<script>
	import { Settings, Moon, Sun, Database, Cpu } from 'lucide-svelte';
	
	let darkMode = true;
	let semanticCache = true;
	let cacheSize = 1000;
	let autoQuantize = true;
	let defaultTemp = 0.7;
	let maxTokens = 2048;
</script>

<svelte:head>
	<title>zllm - Settings</title>
</svelte:head>

<div class="h-full p-6 overflow-y-auto">
	<h1 class="text-2xl font-bold text-white mb-6">Settings</h1>
	
	<div class="max-w-2xl space-y-6">
		<!-- Appearance -->
		<div class="bg-dark-200 rounded-lg p-6">
			<h2 class="text-lg font-semibold text-white mb-4">Appearance</h2>
			
			<div class="flex items-center justify-between">
				<div>
					<p class="text-white">Dark Mode</p>
					<p class="text-sm text-gray-400">Use dark theme for the interface</p>
				</div>
				<button
					on:click={() => darkMode = !darkMode}
					class="w-12 h-6 rounded-full transition-colors {darkMode ? 'bg-primary-500' : 'bg-gray-600'}"
				>
					<div class="w-5 h-5 rounded-full bg-white transform transition-transform 
											{darkMode ? 'translate-x-6' : 'translate-x-0.5'}"></div>
				</button>
			</div>
		</div>
		
		<!-- Cache Settings -->
		<div class="bg-dark-200 rounded-lg p-6">
			<h2 class="text-lg font-semibold text-white mb-4">Cache Settings</h2>
			
			<div class="space-y-4">
				<div class="flex items-center justify-between">
					<div>
						<p class="text-white">Semantic Cache</p>
						<p class="text-sm text-gray-400">Cache similar prompts using AI</p>
					</div>
					<button
						on:click={() => semanticCache = !semanticCache}
						class="w-12 h-6 rounded-full transition-colors {semanticCache ? 'bg-primary-500' : 'bg-gray-600'}"
					>
						<div class="w-5 h-5 rounded-full bg-white transform transition-transform 
												{semanticCache ? 'translate-x-6' : 'translate-x-0.5'}"></div>
					</button>
				</div>
				
				<div>
					<label class="block text-white mb-1">Cache Size</label>
					<p class="text-sm text-gray-400 mb-2">Maximum number of cached responses</p>
					<input
						bind:value={cacheSize}
						type="number"
						class="w-full bg-dark-300 border border-dark-100 rounded-lg px-3 py-2 
									 text-gray-200 outline-none focus:border-primary-500"
					/>
				</div>
			</div>
		</div>
		
		<!-- Model Settings -->
		<div class="bg-dark-200 rounded-lg p-6">
			<h2 class="text-lg font-semibold text-white mb-4">Model Settings</h2>
			
			<div class="space-y-4">
				<div class="flex items-center justify-between">
					<div>
						<p class="text-white">Auto-Quantize</p>
						<p class="text-sm text-gray-400">Automatically use quantization when needed</p>
					</div>
					<button
						on:click={() => autoQuantize = !autoQuantize}
						class="w-12 h-6 rounded-full transition-colors {autoQuantize ? 'bg-primary-500' : 'bg-gray-600'}"
					>
						<div class="w-5 h-5 rounded-full bg-white transform transition-transform 
												{autoQuantize ? 'translate-x-6' : 'translate-x-0.5'}"></div>
					</button>
				</div>
				
				<div>
					<label class="block text-white mb-1">Default Temperature</label>
					<p class="text-sm text-gray-400 mb-2">Controls randomness (0.0 - 2.0)</p>
					<input
						bind:value={defaultTemp}
						type="number"
						step="0.1"
						min="0"
						max="2"
						class="w-full bg-dark-300 border border-dark-100 rounded-lg px-3 py-2 
									 text-gray-200 outline-none focus:border-primary-500"
					/>
				</div>
				
				<div>
					<label class="block text-white mb-1">Max Tokens</label>
					<p class="text-sm text-gray-400 mb-2">Maximum response length</p>
					<input
						bind:value={maxTokens}
						type="number"
						min="1"
						max="32000"
						class="w-full bg-dark-300 border border-dark-100 rounded-lg px-3 py-2 
									 text-gray-200 outline-none focus:border-primary-500"
					/>
				</div>
			</div>
		</div>
		
		<!-- Save Button -->
		<div class="flex justify-end">
			<button class="px-6 py-2 bg-primary-600 hover:bg-primary-500 text-white rounded-lg">
				Save Settings
			</button>
		</div>
	</div>
</div>
