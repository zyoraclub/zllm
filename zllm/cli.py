"""
zllm Command Line Interface.

Provides both end-user friendly interactive mode and developer CLI commands.
"""

import sys
import time
import threading
from pathlib import Path
from typing import Optional
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich import print as rprint

console = Console()


def format_bytes(bytes_val: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f}TB"


def get_memory_bar(used: float, total: float, width: int = 20) -> str:
    """Create a visual memory usage bar."""
    if total == 0:
        return "[dim]N/A[/dim]"
    pct = min(used / total, 1.0)
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    color = "green" if pct < 0.6 else "yellow" if pct < 0.85 else "red"
    return f"[{color}]{bar}[/{color}] {pct*100:.0f}%"


@click.group(invoke_without_command=True)
@click.option("--version", is_flag=True, help="Show version")
@click.pass_context
def main(ctx, version):
    """
    zllm - Memory-efficient LLM inference for everyone.
    
    Run without arguments for interactive mode, or use commands for CLI mode.
    """
    if version:
        from zllm import __version__
        console.print(f"zllm version {__version__}")
        return
    
    if ctx.invoked_subcommand is None:
        # Interactive mode
        interactive_mode()


def interactive_mode():
    """Launch the interactive user-friendly mode."""
    console.print(Panel.fit(
        "[bold cyan]🤖 Welcome to zllm![/bold cyan]\n\n"
        "Memory-efficient LLM inference for everyone.",
        border_style="cyan"
    ))
    
    console.print("\n[bold]What would you like to do?[/bold]\n")
    
    options = [
        ("1", "💬 Start chatting", "chat"),
        ("2", "📥 Download a model", "pull"),
        ("3", "📋 List downloaded models", "list"),
        ("4", "🌐 Start API server", "serve"),
        ("5", "🖥️  Check system info", "info"),
        ("6", "⚙️  Settings", "config"),
        ("0", "Exit", "exit"),
    ]
    
    for key, label, _ in options:
        console.print(f"  [{key}] {label}")
    
    console.print()
    choice = Prompt.ask("Enter choice", choices=["0", "1", "2", "3", "4", "5", "6"], default="1")
    
    action = next((opt[2] for opt in options if opt[0] == choice), "exit")
    
    if action == "chat":
        interactive_chat()
    elif action == "pull":
        interactive_pull()
    elif action == "list":
        list_models()
    elif action == "serve":
        model = Prompt.ask("Model to serve", default="")
        if model:
            serve.callback(model=model, host="127.0.0.1", port=8000, reload=False)
    elif action == "info":
        info.callback()
    elif action == "config":
        console.print("[yellow]Settings coming soon![/yellow]")
    elif action == "exit":
        console.print("[cyan]Goodbye! 👋[/cyan]")
        sys.exit(0)


def interactive_chat():
    """Interactive chat mode."""
    from zllm.models.hub import ModelHub
    from zllm.hardware.auto_detect import detect_hardware
    
    hub = ModelHub()
    hw = detect_hardware()
    
    # Get recommended models
    max_memory = hw.gpus[0].free_memory_gb if hw.has_gpu else hw.system.available_ram_gb
    recommended = hub.get_recommended(max_memory)
    
    console.print("\n[bold]Select a model:[/bold]\n")
    
    for i, model_id in enumerate(recommended[:5], 1):
        name = model_id.split("/")[-1]
        console.print(f"  [{i}] {name}")
    
    console.print(f"  [0] Enter custom model ID")
    console.print()
    
    choice = Prompt.ask("Choice", default="1")
    
    if choice == "0":
        model_id = Prompt.ask("Enter HuggingFace model ID")
    else:
        try:
            idx = int(choice) - 1
            model_id = recommended[idx]
        except (ValueError, IndexError):
            model_id = recommended[0]
    
    console.print(f"\n[cyan]Loading {model_id}...[/cyan]\n")
    
    # Start chat
    run_cmd.callback(model=model_id, prompt=None, system=None, stream=True)


def interactive_pull():
    """Interactive model download."""
    model_id = Prompt.ask("Enter HuggingFace model ID (e.g., meta-llama/Llama-3-8B-Instruct)")
    if model_id:
        pull.callback(model_id=model_id)


# ============== CLI Commands ==============

@main.command()
@click.argument("model")
@click.option("--prompt", "-p", help="Single prompt to run (non-interactive)")
@click.option("--system", "-s", help="System prompt")
@click.option("--stream/--no-stream", default=True, help="Stream output")
@click.option("--speed", type=click.Choice(["fast", "balanced", "memory"]), default="balanced",
              help="Speed vs memory trade-off (fast=more VRAM, memory=less VRAM)")
@click.option("--speculative", "-sp", default=None, help="Enable speculative decoding with draft model")
def run(model: str, prompt: Optional[str], system: Optional[str], stream: bool, speed: str, speculative: Optional[str]):
    """
    Run a model for chat.
    
    Speed modes:
      fast     - Maximum speed, uses 75% of available VRAM
      balanced - Sweet spot, fast + efficient (default)
      memory   - Minimum VRAM, slower
    
    Speculative Decoding:
      Use --speculative to enable 2-3x faster inference with a draft model.
      The draft model should be a smaller variant (e.g., 7B for 70B target).
    
    Examples:
        zllm run llama3
        zllm run meta-llama/Llama-3-8B-Instruct --speed fast
        zllm run ./local-model --prompt "Hello!" --speed memory
        zllm run llama3-70b --speculative llama3  # 2-3x faster!
    """
    run_cmd.callback(model, prompt, system, stream, speed, speculative)


def run_cmd_callback(model: str, prompt: Optional[str], system: Optional[str], stream: bool, speed: str = "balanced", speculative: Optional[str] = None):
    """Implementation of the run command."""
    from zllm import ZLLM, ZLLMConfig
    
    # Resolve model aliases
    model_aliases = {
        "llama3": "meta-llama/Llama-3-8B-Instruct",
        "llama3-70b": "meta-llama/Llama-3-70B-Instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
        "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "qwen": "Qwen/Qwen2-7B-Instruct",
        "phi": "microsoft/Phi-3-mini-4k-instruct",
        "gemma": "google/gemma-2-9b-it",
    }
    
    model_id = model_aliases.get(model.lower(), model)
    draft_model_id = model_aliases.get(speculative.lower(), speculative) if speculative else None
    
    # Configure with speculative decoding if requested
    config = ZLLMConfig(
        model_id=model_id, 
        speed_mode=speed,
        enable_speculative=draft_model_id is not None,
        draft_model_id=draft_model_id,
    )
    
    try:
        llm = ZLLM(model_id, config=config)
    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        return
    
    if prompt:
        # Single prompt mode
        if stream:
            for token in llm.chat_stream(prompt, system_prompt=system):
                console.print(token, end="")
            console.print()
        else:
            response = llm.chat(prompt, system_prompt=system)
            console.print(response)
    else:
        # Interactive chat mode
        spec_info = f"\n[dim]Speculative: {draft_model_id}[/dim]" if draft_model_id else ""
        console.print(Panel(
            f"[green]Model loaded: {model_id}[/green]\n"
            f"[dim]Speed mode: {llm.memory_manager.speed_mode.value if llm.memory_manager else speed}[/dim]{spec_info}\n\n"
            "Type your message and press Enter.\n"
            "Commands: /help, /memory, /efficiency, /speed, /auto, /upgrade, /exit\",
            title="💬 Chat Mode",
            border_style="green"
        ))
        
        # Check if upgrade is available and show hint
        try:
            upgrade_info = llm.can_upgrade()
            if upgrade_info.get("can_upgrade"):
                console.print(f"\n[cyan]💡 Performance upgrade available![/cyan]")
                console.print(f"   Current: {upgrade_info['current_quantization']} → Recommended: {upgrade_info['recommended_quantization'] or 'fp16'}")
                console.print(f"   Expected speedup: {upgrade_info['estimated_speedup']}")
                console.print("[dim]   Type /upgrade for details[/dim]\n")
        except Exception:
            pass  # Silently ignore upgrade check errors
        
        history = []
        tokens_generated = 0
        start_time = time.time()
        
        while True:
            try:
                user_input = Prompt.ask("\n[bold blue]You[/bold blue]")
                
                if user_input.lower() in ["exit", "quit", "q", "/exit", "/quit", "/q"]:
                    # Show session stats before exit
                    elapsed = time.time() - start_time
                    console.print(f"\n[dim]Session: {len(history)//2} exchanges, {tokens_generated} tokens, {elapsed:.0f}s[/dim]")
                    console.print("[cyan]Goodbye! 👋[/cyan]")
                    break
                
                if user_input.lower() == "/help":
                    help_text = """
[bold cyan]Available Commands:[/bold cyan]

  [green]/help[/green]        - Show this help message
  [green]/memory[/green]      - Show VRAM/memory usage
  [green]/efficiency[/green]  - Check GPU efficiency & get recommendations
  [green]/stats[/green]       - Show cache and generation stats
  [green]/kv[/green]          - Show KV cache statistics  
  [green]/clear[/green]       - Clear conversation history
  [green]/speed[/green]       - Show/change speed mode (fast/balanced/memory)
  [green]/auto[/green]        - Enable/disable auto speed adjustment
  [green]/quiet[/green]       - Hide optimization suggestions
  [green]/upgrade[/green]     - Check/perform model upgrade for better speed
  [green]/speculative[/green] - Show speculative decoding stats
  [green]/exit[/green]        - Exit chat

[bold cyan]Performance Commands:[/bold cyan]
  /speed fast      - Maximize speed (uses more VRAM)
  /speed memory    - Minimize memory (slower but fits larger contexts)
  /auto on         - Auto-adjust speed based on GPU usage
  /upgrade now     - Reload model without quantization (fastest)
  /efficiency      - Show current GPU utilization & suggestions
"""
                    console.print(help_text)
                    continue
                
                if user_input.lower() == "/clear":
                    history = []
                    console.print("[yellow]✓ History cleared[/yellow]")
                    continue
                
                if user_input.lower() == "/stats":
                    stats = llm.get_cache_stats() if hasattr(llm, 'get_cache_stats') else {}
                    elapsed = time.time() - start_time
                    
                    stats_table = Table(title="📊 Session Statistics", box=None)
                    stats_table.add_column("Metric", style="cyan")
                    stats_table.add_column("Value", style="green")
                    
                    stats_table.add_row("Messages", str(len(history) // 2))
                    stats_table.add_row("Tokens Generated", str(tokens_generated))
                    stats_table.add_row("Session Duration", f"{elapsed:.0f}s")
                    if tokens_generated > 0 and elapsed > 0:
                        stats_table.add_row("Avg Tokens/sec", f"{tokens_generated/elapsed:.1f}")
                    if stats:
                        stats_table.add_row("Cache Hits", str(stats.get('hits', 0)))
                        stats_table.add_row("Cache Misses", str(stats.get('misses', 0)))
                    
                    console.print(stats_table)
                    continue
                
                if user_input.lower() == "/memory":
                    try:
                        import torch
                        if torch.cuda.is_available():
                            allocated = torch.cuda.memory_allocated() / 1024**3
                            reserved = torch.cuda.memory_reserved() / 1024**3
                            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                            console.print(f"\n[bold]GPU Memory:[/bold]")
                            console.print(f"  Allocated: {allocated:.2f}GB")
                            console.print(f"  Reserved:  {reserved:.2f}GB")
                            console.print(f"  Total:     {total:.2f}GB")
                            console.print(f"  Usage:     {get_memory_bar(allocated, total)}")
                        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                            # MPS doesn't have detailed memory tracking
                            console.print("\n[bold]Apple Silicon GPU[/bold]: Active")
                            console.print("[dim]Detailed memory stats not available for MPS[/dim]")
                        else:
                            import psutil
                            mem = psutil.virtual_memory()
                            console.print(f"\n[bold]System Memory:[/bold]")
                            console.print(f"  Used:  {mem.used/1024**3:.2f}GB")
                            console.print(f"  Total: {mem.total/1024**3:.2f}GB")
                            console.print(f"  Usage: {get_memory_bar(mem.used, mem.total)}")
                    except Exception as e:
                        console.print(f"[yellow]Could not get memory stats: {e}[/yellow]")
                    continue
                
                if user_input.lower() == "/kv":
                    if hasattr(llm, 'kv_cache_manager') and llm.kv_cache_manager:
                        kv_stats = llm.kv_cache_manager.get_stats()
                        console.print("\n[bold cyan]KV Cache Stats:[/bold cyan]")
                        console.print(f"  Quantization: {kv_stats.get('quantization', 'fp16')}")
                        console.print(f"  Prompt Cache Hits: {kv_stats.get('prompt_cache_hits', 0)}")
                        console.print(f"  Memory Saved: {format_bytes(kv_stats.get('memory_saved', 0))}")
                    else:
                        console.print("[dim]KV Cache manager not initialized[/dim]")
                    continue
                
                if user_input.lower().startswith("/speed"):
                    parts = user_input.split()
                    if len(parts) == 1:
                        current = llm.memory_manager.speed_mode.value if llm.memory_manager else "unknown"
                        console.print(f"\n[bold]Current speed mode:[/bold] {current}")
                        console.print("[dim]Usage: /speed <fast|balanced|memory>[/dim]")
                        
                        # Check if upgrade is available
                        upgrade_info = llm.can_upgrade()
                        if upgrade_info.get("can_upgrade"):
                            console.print(f"\n[green]💡 Performance upgrade available![/green]")
                            console.print(f"   {upgrade_info['reason']}")
                            console.print(f"   Expected speedup: {upgrade_info['estimated_speedup']}")
                            console.print("[dim]   Use /upgrade to reload with better performance[/dim]")
                    else:
                        new_mode = parts[1].lower()
                        if new_mode in ["fast", "balanced", "memory"]:
                            if llm.memory_manager:
                                from zllm.core.memory import SpeedMode
                                llm.memory_manager.speed_mode = SpeedMode(new_mode)
                                console.print(f"[green]✓ Speed mode changed to: {new_mode}[/green]")
                                
                                # Show upgrade hint if available
                                upgrade_info = llm.can_upgrade()
                                if upgrade_info.get("can_upgrade") and new_mode == "fast":
                                    console.print(f"\n[cyan]💡 Tip: Model was loaded with {upgrade_info['current_quantization']} quantization.[/cyan]")
                                    console.print(f"   For best 'fast' mode performance, use [bold]/upgrade[/bold] to reload without quantization.")
                                    console.print(f"   Expected speedup: {upgrade_info['estimated_speedup']}")
                        else:
                            console.print("[red]Invalid mode. Use: fast, balanced, or memory[/red]")
                    continue
                
                if user_input.lower().startswith("/upgrade"):
                    upgrade_info = llm.can_upgrade()
                    parts = user_input.split()
                    
                    if len(parts) == 1:
                        # Show upgrade status
                        console.print("\n[bold cyan]Model Upgrade Status:[/bold cyan]")
                        console.print(f"  Current quantization: {upgrade_info['current_quantization'] or 'fp16 (none)'}")
                        console.print(f"  Memory available: {upgrade_info['memory_available_gb']:.1f}GB")
                        
                        if upgrade_info["can_upgrade"]:
                            target = upgrade_info['recommended_quantization'] or 'fp16'
                            console.print(f"\n[green]✓ Upgrade available![/green]")
                            console.print(f"  Target: {target}")
                            console.print(f"  Additional VRAM needed: {upgrade_info['memory_required_gb']:.1f}GB")
                            console.print(f"  Expected speedup: {upgrade_info['estimated_speedup']}")
                            console.print("\n[dim]Use '/upgrade now' to reload with better performance[/dim]")
                            console.print("[dim]Use '/upgrade fp16' or '/upgrade int8' for specific target[/dim]")
                        else:
                            console.print(f"\n[yellow]ℹ️  {upgrade_info['reason']}[/yellow]")
                    
                    elif parts[1].lower() == "now":
                        if not upgrade_info["can_upgrade"]:
                            console.print(f"[yellow]Cannot upgrade: {upgrade_info['reason']}[/yellow]")
                        else:
                            console.print("\n[bold]⚠️  This will reload the model. Continue? (y/n)[/bold]")
                            confirm = console.input().strip().lower()
                            if confirm in ["y", "yes"]:
                                try:
                                    llm.upgrade_model()
                                except Exception as e:
                                    console.print(f"[red]Upgrade failed: {e}[/red]")
                            else:
                                console.print("[dim]Upgrade cancelled[/dim]")
                    
                    elif parts[1].lower() in ["fp16", "int8", "int4", "none"]:
                        target = parts[1].lower()
                        if target in ["fp16", "none"]:
                            target = None
                        console.print(f"\n[bold]⚠️  This will reload the model with {target or 'fp16'}. Continue? (y/n)[/bold]")
                        confirm = console.input().strip().lower()
                        if confirm in ["y", "yes"]:
                            try:
                                llm.upgrade_model(target_quantization=target)
                            except Exception as e:
                                console.print(f"[red]Upgrade failed: {e}[/red]")
                        else:
                            console.print("[dim]Upgrade cancelled[/dim]")
                    else:
                        console.print("[dim]Usage: /upgrade [now|fp16|int8|int4][/dim]")
                    continue
                
                if user_input.lower() == "/speculative":
                    if hasattr(llm, 'speculative_decoder') and llm.speculative_decoder:
                        spec_stats = llm.speculative_decoder.get_stats()
                        console.print("\n[bold cyan]Speculative Decoding Stats:[/bold cyan]")
                        console.print(f"  Acceptance Rate: {spec_stats.get('acceptance_rate', 0)*100:.1f}%")
                        console.print(f"  Speedup Factor:  {spec_stats.get('speedup_factor', 1.0):.2f}x")
                        console.print(f"  Accepted Tokens: {spec_stats.get('accepted_tokens', 0)}")
                        console.print(f"  Rejected Tokens: {spec_stats.get('rejected_tokens', 0)}")
                        console.print(f"  Draft Passes:    {spec_stats.get('draft_forward_passes', 0)}")
                        console.print(f"  Target Passes:   {spec_stats.get('target_forward_passes', 0)}")
                        if spec_stats.get('is_fallback_mode'):
                            console.print("[yellow]  ⚠️  In fallback mode (low acceptance rate)[/yellow]")
                    else:
                        console.print("[dim]Speculative decoding not enabled.[/dim]")
                        console.print("[dim]Start with: zllm run model --speculative draft-model[/dim]")
                    continue
                
                if user_input.lower().startswith("/auto"):
                    parts = user_input.split()
                    if len(parts) == 1:
                        # Show auto status
                        auto_enabled = llm._runtime_monitor.get("auto_adjust", False) if hasattr(llm, '_runtime_monitor') else False
                        console.print(f"\n[bold]Auto-adjust:[/bold] {'enabled' if auto_enabled else 'disabled'}")
                        console.print("[dim]Usage: /auto on|off[/dim]")
                        console.print("[dim]When enabled, engine automatically adjusts speed based on memory[/dim]")
                    else:
                        setting = parts[1].lower()
                        if setting in ["on", "true", "yes", "1"]:
                            llm.set_auto_adjust(True)
                            console.print("[green]✓ Auto-adjust enabled[/green]")
                            console.print("[dim]Engine will now automatically optimize speed based on GPU memory[/dim]")
                            # Show current status
                            mem_check = llm.check_runtime_memory()
                            console.print(f"  Current GPU: {mem_check.get('usage_percent', 0):.0f}% ({mem_check.get('allocated_gb', 0):.1f}GB / {mem_check.get('total_gb', 0):.1f}GB)")
                            console.print(f"  Speed mode: {mem_check.get('current_speed_mode', 'unknown')}")
                        elif setting in ["off", "false", "no", "0"]:
                            llm.set_auto_adjust(False)
                            console.print("[yellow]✓ Auto-adjust disabled[/yellow]")
                        else:
                            console.print("[red]Invalid option. Use: /auto on or /auto off[/red]")
                    continue
                
                if user_input.lower() == "/quiet":
                    llm.silence_recommendations()
                    console.print("[dim]✓ Optimization suggestions disabled for this session[/dim]")
                    console.print("[dim]Use /auto on for automatic optimization[/dim]")
                    continue
                
                if user_input.lower() == "/efficiency":
                    mem_check = llm.check_runtime_memory()
                    console.print("\n[bold cyan]GPU Efficiency Status:[/bold cyan]")
                    console.print(f"  GPU Usage: {mem_check.get('usage_percent', 0):.0f}%")
                    console.print(f"  Allocated: {mem_check.get('allocated_gb', 0):.1f}GB / {mem_check.get('total_gb', 0):.1f}GB")
                    console.print(f"  Free: {mem_check.get('free_gb', 0):.1f}GB")
                    console.print(f"  Speed Mode: {mem_check.get('current_speed_mode', 'unknown')}")
                    console.print(f"  Status: {mem_check.get('status', 'unknown')}")
                    
                    if mem_check.get("recommendation"):
                        console.print(f"\n[cyan]💡 {mem_check['recommendation']}[/cyan]")
                        if mem_check.get("can_upgrade"):
                            console.print("[dim]   Run /upgrade now to apply[/dim]")
                        elif mem_check.get("can_speed_up"):
                            console.print("[dim]   Run /speed fast or /auto on to apply[/dim]")
                    else:
                        console.print("\n[green]✓ Running at optimal efficiency[/green]")
                    continue
                
                # Generate response
                console.print("\n[bold green]Assistant[/bold green]:", end=" ")
                
                full_response = []
                gen_start = time.time()
                token_count = 0
                
                if stream:
                    for token in llm.chat_stream(user_input, system_prompt=system, history=history):
                        console.print(token, end="")
                        full_response.append(token)
                        token_count += 1
                    console.print()
                else:
                    response = llm.chat(user_input, system_prompt=system, history=history)
                    console.print(response)
                    full_response.append(response)
                    token_count = len(response.split())  # Approximate
                
                gen_time = time.time() - gen_start
                tokens_generated += token_count
                
                # Show generation stats
                if gen_time > 0:
                    console.print(f"[dim]{token_count} tokens in {gen_time:.1f}s ({token_count/gen_time:.1f} tok/s)[/dim]")
                
                # Check for runtime optimization recommendations
                try:
                    recommendation = llm.get_speed_recommendation()
                    if recommendation and recommendation.get("recommendation"):
                        console.print()
                        if recommendation.get("action") == "upgrade":
                            console.print(f"[cyan]💡 {recommendation['recommendation']}[/cyan]")
                        elif recommendation.get("action") == "speed_up":
                            console.print(f"[cyan]💡 {recommendation['recommendation']}[/cyan]")
                            console.print("[dim]   Use /speed fast or /auto to optimize automatically[/dim]")
                        elif recommendation.get("action") == "slow_down":
                            console.print(f"[yellow]⚠️  {recommendation['recommendation']}[/yellow]")
                        console.print("[dim]   Use /quiet to hide these suggestions[/dim]")
                except Exception:
                    pass  # Don't let monitoring errors interrupt chat
                
                # Update history
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": "".join(full_response)})
                
            except KeyboardInterrupt:
                console.print("\n[cyan]Goodbye! 👋[/cyan]")
                break
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")

# Create a helper object for the callback
class run_cmd:
    callback = staticmethod(run_cmd_callback)


@main.command()
@click.argument("model_id")
def pull(model_id: str):
    """
    Download a model from HuggingFace Hub.
    
    Examples:
        zllm pull meta-llama/Llama-3-8B-Instruct
        zllm pull mistralai/Mistral-7B-Instruct-v0.3
    """
    from zllm.models.loader import ModelLoader
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    
    loader = ModelLoader()
    
    console.print(f"[cyan]Downloading {model_id}...[/cyan]\n")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading...", total=None)
            
            path = loader.download_model(model_id)
            
            progress.update(task, description="Download complete!")
        
        console.print(f"\n[green]✓ Model downloaded to: {path}[/green]")
        
        # Show model info
        info = loader.get_model_info(model_id)
        console.print(f"  Type: {info.model_type}")
        console.print(f"  Parameters: {info.params_billions:.1f}B")
        console.print(f"  Size: {info.size_gb:.1f}GB")
        
    except Exception as e:
        console.print(f"[red]Error downloading model: {e}[/red]")


@main.command()
def list():
    """List downloaded models."""
    list_models()


def list_models():
    """Show downloaded models."""
    from zllm.core.config import get_default_data_dir
    from zllm.models.loader import ModelRegistry
    
    data_dir = get_default_data_dir()
    registry = ModelRegistry(data_dir)
    models = registry.list_models()
    
    if not models:
        console.print("[yellow]No models downloaded yet.[/yellow]")
        console.print("Run [cyan]zllm pull <model_id>[/cyan] to download a model.")
        return
    
    table = Table(title="Downloaded Models")
    table.add_column("Model", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Size", style="yellow")
    table.add_column("Parameters", style="magenta")
    
    for model in models:
        table.add_row(
            model["model_id"],
            model.get("model_type", "unknown"),
            f"{model.get('size_gb', 0):.1f}GB",
            f"{model.get('params_billions', 0):.1f}B",
        )
    
    console.print(table)


@main.command()
@click.option("--model", "-m", required=True, help="Model to serve")
@click.option("--host", "-h", default="127.0.0.1", help="Host to bind to")
@click.option("--port", "-p", default=8000, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
@click.option("--speed", type=click.Choice(["fast", "balanced", "memory"]), default="balanced",
              help="Speed vs memory trade-off")
def serve(model: str, host: str, port: int, reload: bool, speed: str):
    """
    Start the API server.
    
    Speed modes:
      fast     - Maximum speed, uses 75% of available VRAM
      balanced - Sweet spot, fast + efficient (default)
      memory   - Minimum VRAM, slower
    
    Examples:
        zllm serve --model llama3
        zllm serve -m meta-llama/Llama-3-8B-Instruct -p 8080 --speed fast
    """
    console.print(Panel(
        f"[green]Starting zllm server[/green]\n\n"
        f"Model: {model}\n"
        f"Speed Mode: {speed}\n"
        f"URL: http://{host}:{port}\n"
        f"API Docs: http://{host}:{port}/docs",
        title="🚀 zllm Server",
        border_style="green"
    ))
    
    import uvicorn
    from zllm.server.api import create_app
    
    app = create_app(model_id=model, speed_mode=speed)
    uvicorn.run(app, host=host, port=port, reload=reload)


@main.command()
def info():
    """Show system information, hardware capabilities, and zllm features."""
    from zllm.hardware.auto_detect import HardwareDetector
    from zllm import __version__
    
    # Header
    console.print(Panel.fit(
        f"[bold cyan]zllm v{__version__}[/bold cyan] - Memory-efficient LLM Inference",
        border_style="cyan"
    ))
    
    detector = HardwareDetector()
    detector.print_summary()
    
    # Show memory recommendations
    hw_info = detector.detect()
    
    console.print("\n[bold]Model Recommendations:[/bold]")
    
    if hw_info.has_gpu:
        gpu_mem = hw_info.gpus[0].free_memory_gb
        console.print(f"\nWith {gpu_mem:.1f}GB GPU memory, you can run:")
        
        if gpu_mem >= 40:
            console.print("  • 70B models at full precision")
            console.print("  • 70B+ models with 4-bit quantization")
        elif gpu_mem >= 16:
            console.print("  • 13B models at full precision")
            console.print("  • 70B models with 4-bit quantization")
        elif gpu_mem >= 8:
            console.print("  • 7B models at full precision")
            console.print("  • 13B models with 4-bit quantization")
        elif gpu_mem >= 4:
            console.print("  • [green]7B models with 4-bit quantization[/green] ← zllm optimized!")
            console.print("  • Smaller models (3B, 1B) at full precision")
        else:
            console.print("  • Smaller models with quantization")
            console.print("  • [green]Layer streaming for larger models[/green] ← zllm feature!")
    else:
        console.print("\n[yellow]No GPU detected. CPU inference will be slower.[/yellow]")
        console.print("Recommended: Use quantized models for better performance.")
    
    # Show zllm-specific features
    console.print("\n" + "─" * 50)
    console.print("\n[bold cyan]⚡ zllm Optimization Features:[/bold cyan]\n")
    
    features_table = Table(show_header=False, box=None, padding=(0, 2))
    features_table.add_column("Feature", style="green")
    features_table.add_column("Status", style="cyan")
    features_table.add_column("Description")
    
    features_table.add_row("🧠 Intelligent Orchestrator", "Active", "Auto-adjusts VRAM usage in real-time")
    features_table.add_row("📍 Hot Layer Pinning", "Active", "Keeps critical layers (12-20) in VRAM")
    features_table.add_row("💾 KV Cache Quantization", "INT8", "50% memory savings on context")
    features_table.add_row("📝 Prompt Caching", "Active", "Instant responses for repeated prompts")
    features_table.add_row("🔄 Continuous Batching", "Active", "2-4x higher server throughput")
    features_table.add_row("⚡ Flash Attention", "Auto", "O(N) memory, 2-4x faster attention")
    features_table.add_row("🚀 Speculative Decoding", "Available", "2-3x speedup with --speculative")
    
    console.print(features_table)
    
    # Speed modes explanation
    console.print("\n[bold cyan]🚀 Speed Modes:[/bold cyan]\n")
    
    speed_table = Table(show_header=True, box=None)
    speed_table.add_column("Mode", style="bold")
    speed_table.add_column("VRAM Usage")
    speed_table.add_column("Best For")
    
    speed_table.add_row("[green]fast[/green]", "75%", "Maximum speed, quick responses")
    speed_table.add_row("[yellow]balanced[/yellow]", "60%", "Default - good speed + efficiency")
    speed_table.add_row("[red]memory[/red]", "40%", "Tight VRAM, larger models")
    
    console.print(speed_table)
    
    # Quick start
    console.print("\n[bold cyan]🎯 Quick Start:[/bold cyan]\n")
    console.print("  [dim]Start chatting:[/dim]     zllm run llama3")
    console.print("  [dim]Memory mode:[/dim]        zllm run mistral --speed memory")
    console.print("  [dim]Speculative (2-3x):[/dim] zllm run llama3-70b --speculative llama3")
    console.print("  [dim]Start server:[/dim]       zllm serve -m llama3")
    console.print("  [dim]Interactive:[/dim]        zllm")
    console.print()


@main.command()
@click.argument("query")
@click.option("--limit", "-n", default=10, help="Number of results")
def search(query: str, limit: int):
    """
    Search for models on HuggingFace Hub.
    
    Examples:
        zllm search llama
        zllm search "code generation" -n 5
    """
    from zllm.models.hub import ModelHub
    
    hub = ModelHub()
    
    console.print(f"[cyan]Searching for '{query}'...[/cyan]\n")
    
    results = hub.search(query, limit=limit)
    
    if not results:
        console.print("[yellow]No models found.[/yellow]")
        return
    
    table = Table(title=f"Search Results: {query}")
    table.add_column("Model", style="cyan")
    table.add_column("Downloads", style="green", justify="right")
    table.add_column("Likes", style="yellow", justify="right")
    
    for model in results:
        table.add_row(
            model.model_id,
            f"{model.downloads:,}",
            f"❤️ {model.likes}",
        )
    
    console.print(table)
    console.print("\nRun [cyan]zllm pull <model_id>[/cyan] to download a model.")


@main.command()
@click.argument("model_path")
@click.option("--bits", "-b", type=click.Choice(["4", "8"]), default="4", help="Quantization bits")
@click.option("--output", "-o", help="Output path")
def quantize(model_path: str, bits: str, output: Optional[str]):
    """
    Quantize a model for reduced memory usage.
    
    Examples:
        zllm quantize ./llama-3-8b --bits 4
        zllm quantize meta-llama/Llama-3-70B -b 4 -o ./llama3-70b-int4
    """
    console.print(f"[cyan]Quantizing model to {bits}-bit...[/cyan]")
    console.print("[yellow]Quantization feature coming soon![/yellow]")
    # TODO: Implement quantization


@main.command()
def ui():
    """Launch the web UI."""
    console.print("[cyan]Starting zllm Web UI...[/cyan]")
    console.print("[yellow]Web UI coming soon![/yellow]")
    # TODO: Launch web UI


@main.command()
@click.argument("model")
@click.option("--prompts", "-n", default=5, help="Number of prompts to run")
@click.option("--max-tokens", "-t", default=100, help="Max tokens per response")
@click.option("--speed", type=click.Choice(["fast", "balanced", "memory"]), default="balanced")
def benchmark(model: str, prompts: int, max_tokens: int, speed: str):
    """
    Benchmark model performance.
    
    Examples:
        zllm benchmark llama3
        zllm benchmark mistral --prompts 10 --speed fast
    """
    from zllm import ZLLM, ZLLMConfig
    import time
    
    test_prompts = [
        "What is 2 + 2?",
        "Write a haiku about coding.",
        "Explain recursion in one sentence.",
        "What is the capital of France?",
        "List 3 programming languages.",
        "What color is the sky?",
        "Who wrote Romeo and Juliet?",
        "What is machine learning?",
        "Name a planet in our solar system.",
        "What is 10 * 10?",
    ]
    
    # Resolve model alias
    model_aliases = {
        "llama3": "meta-llama/Llama-3-8B-Instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
        "phi": "microsoft/Phi-3-mini-4k-instruct",
    }
    model_id = model_aliases.get(model.lower(), model)
    
    console.print(Panel(
        f"[bold]Benchmarking: {model_id}[/bold]\n"
        f"Prompts: {prompts} | Max tokens: {max_tokens} | Speed: {speed}",
        title="🏃 Benchmark",
        border_style="cyan"
    ))
    
    config = ZLLMConfig(model_id=model_id, speed_mode=speed, max_new_tokens=max_tokens)
    
    try:
        llm = ZLLM(model_id, config=config)
    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        return
    
    results = []
    total_tokens = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running benchmark...", total=prompts)
        
        for i in range(prompts):
            prompt = test_prompts[i % len(test_prompts)]
            
            start = time.perf_counter()
            response = llm.chat(prompt)
            elapsed = time.perf_counter() - start
            
            tokens = len(response.split())  # Approximate
            total_tokens += tokens
            results.append({
                "prompt": prompt[:30] + "..." if len(prompt) > 30 else prompt,
                "tokens": tokens,
                "time": elapsed,
                "tps": tokens / elapsed if elapsed > 0 else 0,
            })
            
            progress.update(task, advance=1)
    
    # Results
    console.print("\n[bold]Results:[/bold]\n")
    
    results_table = Table(title="Benchmark Results")
    results_table.add_column("#", style="dim")
    results_table.add_column("Prompt", style="cyan")
    results_table.add_column("Tokens", justify="right")
    results_table.add_column("Time", justify="right")
    results_table.add_column("Tok/s", justify="right", style="green")
    
    for i, r in enumerate(results, 1):
        results_table.add_row(
            str(i),
            r["prompt"],
            str(r["tokens"]),
            f"{r['time']:.2f}s",
            f"{r['tps']:.1f}",
        )
    
    console.print(results_table)
    
    # Summary
    total_time = sum(r["time"] for r in results)
    avg_tps = total_tokens / total_time if total_time > 0 else 0
    
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Total prompts: {prompts}")
    console.print(f"  Total tokens:  {total_tokens}")
    console.print(f"  Total time:    {total_time:.2f}s")
    console.print(f"  [green]Avg tokens/sec: {avg_tps:.1f}[/green]")
    
    # Memory stats
    memory_stats = llm.get_memory_stats()
    console.print(f"\n[bold]Memory:[/bold]")
    console.print(f"  Speed mode: {memory_stats.get('speed_mode', 'unknown')}")
    if 'gpu_allocated_gb' in memory_stats:
        console.print(f"  GPU allocated: {memory_stats['gpu_allocated_gb']:.2f}GB")
    
    llm.unload()
    console.print("\n[green]✓ Benchmark complete![/green]")


@main.command()
def status():
    """Show system status and running processes."""
    from zllm.hardware.auto_detect import detect_hardware
    import psutil
    
    hw = detect_hardware()
    
    console.print(Panel.fit("[bold cyan]zllm Status[/bold cyan]", border_style="cyan"))
    
    # System
    console.print("\n[bold]System:[/bold]")
    mem = psutil.virtual_memory()
    console.print(f"  RAM: {get_memory_bar(mem.used, mem.total)} ({mem.used/1024**3:.1f}/{mem.total/1024**3:.1f}GB)")
    console.print(f"  CPU: {psutil.cpu_percent():.0f}% usage")
    
    # GPU
    if hw.has_gpu:
        gpu = hw.gpus[0]
        console.print(f"\n[bold]GPU ({gpu.name}):[/bold]")
        used = gpu.total_memory_gb - gpu.free_memory_gb
        console.print(f"  VRAM: {get_memory_bar(used, gpu.total_memory_gb)} ({used:.1f}/{gpu.total_memory_gb:.1f}GB)")
    
    # Processes
    console.print("\n[bold]Python Processes:[/bold]")
    count = 0
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            if 'python' in proc.info['name'].lower():
                mem_mb = proc.info['memory_info'].rss / 1024**2
                console.print(f"  PID {proc.info['pid']}: {mem_mb:.0f}MB")
                count += 1
                if count >= 5:
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    console.print()


if __name__ == "__main__":
    main()
