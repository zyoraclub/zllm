<script>
	import { onMount } from 'svelte';
	import { Download, Trash2, HardDrive, Loader } from 'lucide-svelte';
	
	let models = [];
	let loading = true;
	let downloading = null;
	let searchQuery = '';
	let searchResults = [];
	let searching = false;
	
	onMount(async () => {
		await loadModels();
	});
	
	async function loadModels() {
		loading = true;
		try {
			const res = await fetch('/v1/models');
			if (res.ok) {
				const data = await res.json();
				models = data.data || [];
			}
		} catch (e) {
			console.error('Failed to load models:', e);
		}
		loading = false;
	}
	
	async function searchModels() {
		if (!searchQuery.trim()) return;
		searching = true;
		// TODO: Implement search API
		searchResults = [];
		searching = false;
	}
</script>

<svelte:head>
	<title>zllm - Models</title>
</svelte:head>

<div class="h-full p-6 overflow-y-auto">
	<h1 class="text-2xl font-bold text-white mb-6">Models</h1>
	
	<!-- Search -->
	<div class="mb-6">
		<div class="flex gap-3">
			<input
				bind:value={searchQuery}
				on:keydown={(e) => e.key === 'Enter' && searchModels()}
				type="text"
				placeholder="Search HuggingFace models..."
				class="flex-1 bg-dark-200 border border-dark-100 rounded-lg px-4 py-2 
							 text-gray-200 placeholder-gray-500 outline-none focus:border-primary-500"
			/>
			<button
				on:click={searchModels}
				class="px-4 py-2 bg-primary-600 hover:bg-primary-500 text-white rounded-lg"
			>
				Search
			</button>
		</div>
	</div>
	
	<!-- Downloaded Models -->
	<div class="mb-8">
		<h2 class="text-lg font-semibold text-white mb-4">Downloaded Models</h2>
		
		{#if loading}
			<div class="flex items-center gap-2 text-gray-400">
				<Loader size={18} class="animate-spin" />
				<span>Loading models...</span>
			</div>
		{:else if models.length === 0}
			<div class="bg-dark-200 rounded-lg p-6 text-center">
				<HardDrive size={48} class="text-gray-500 mx-auto mb-3" />
				<p class="text-gray-400">No models downloaded yet.</p>
				<p class="text-gray-500 text-sm mt-1">Search and download a model to get started.</p>
			</div>
		{:else}
			<div class="grid gap-4">
				{#each models as model}
					<div class="bg-dark-200 rounded-lg p-4 flex items-center justify-between">
						<div>
							<h3 class="font-medium text-white">{model.id}</h3>
							<p class="text-sm text-gray-400">Ready to use</p>
						</div>
						<div class="flex items-center gap-2">
							<button class="p-2 hover:bg-dark-100 rounded-lg text-red-400">
								<Trash2 size={18} />
							</button>
						</div>
					</div>
				{/each}
			</div>
		{/if}
	</div>
	
	<!-- Recommended Models -->
	<div>
		<h2 class="text-lg font-semibold text-white mb-4">Recommended Models</h2>
		<div class="grid gap-4">
			{#each ['meta-llama/Llama-3-8B-Instruct', 'mistralai/Mistral-7B-Instruct-v0.3', 'Qwen/Qwen2-7B-Instruct'] as model}
				<div class="bg-dark-200 rounded-lg p-4 flex items-center justify-between">
					<div>
						<h3 class="font-medium text-white">{model}</h3>
						<p class="text-sm text-gray-400">~14GB download</p>
					</div>
					<button 
						class="flex items-center gap-2 px-3 py-1.5 bg-primary-600 hover:bg-primary-500 
									 text-white rounded-lg text-sm"
					>
						<Download size={16} />
						<span>Download</span>
					</button>
				</div>
			{/each}
		</div>
	</div>
</div>
