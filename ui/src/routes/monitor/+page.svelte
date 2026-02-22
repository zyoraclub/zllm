<script>
	import { onMount } from 'svelte';
	import { Activity, Cpu, HardDrive, Zap, Database } from 'lucide-svelte';
	
	let systemInfo = null;
	let cacheStats = null;
	let healthStatus = null;
	
	onMount(async () => {
		await Promise.all([
			loadSystemInfo(),
			loadCacheStats(),
			loadHealth(),
		]);
		
		// Refresh every 5 seconds
		const interval = setInterval(async () => {
			await Promise.all([
				loadSystemInfo(),
				loadCacheStats(),
			]);
		}, 5000);
		
		return () => clearInterval(interval);
	});
	
	async function loadSystemInfo() {
		try {
			const res = await fetch('/api/system');
			if (res.ok) {
				systemInfo = await res.json();
			}
		} catch (e) {
			console.error('Failed to load system info:', e);
		}
	}
	
	async function loadCacheStats() {
		try {
			const res = await fetch('/api/cache/stats');
			if (res.ok) {
				cacheStats = await res.json();
			}
		} catch (e) {
			// Cache stats may not be available if no model loaded
		}
	}
	
	async function loadHealth() {
		try {
			const res = await fetch('/health');
			if (res.ok) {
				healthStatus = await res.json();
			}
		} catch (e) {
			console.error('Failed to load health:', e);
		}
	}
	
	async function clearCache() {
		try {
			await fetch('/api/cache/clear', { method: 'POST' });
			await loadCacheStats();
		} catch (e) {
			console.error('Failed to clear cache:', e);
		}
	}
</script>

<svelte:head>
	<title>zllm - Monitor</title>
</svelte:head>

<div class="h-full p-6 overflow-y-auto">
	<h1 class="text-2xl font-bold text-white mb-6">System Monitor</h1>
	
	<!-- Status Cards -->
	<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
		<!-- Status -->
		<div class="bg-dark-200 rounded-lg p-4">
			<div class="flex items-center gap-3 mb-2">
				<Activity size={20} class="text-green-400" />
				<span class="text-gray-400">Status</span>
			</div>
			<p class="text-2xl font-bold text-white">
				{healthStatus?.status === 'healthy' ? 'Online' : 'Offline'}
			</p>
		</div>
		
		<!-- RAM -->
		<div class="bg-dark-200 rounded-lg p-4">
			<div class="flex items-center gap-3 mb-2">
				<Cpu size={20} class="text-blue-400" />
				<span class="text-gray-400">RAM Usage</span>
			</div>
			<p class="text-2xl font-bold text-white">
				{#if systemInfo}
					{systemInfo.ram_available_gb?.toFixed(1)}GB
					<span class="text-sm text-gray-400">/ {systemInfo.ram_total_gb?.toFixed(1)}GB</span>
				{:else}
					--
				{/if}
			</p>
		</div>
		
		<!-- VRAM -->
		<div class="bg-dark-200 rounded-lg p-4">
			<div class="flex items-center gap-3 mb-2">
				<HardDrive size={20} class="text-purple-400" />
				<span class="text-gray-400">VRAM Usage</span>
			</div>
			<p class="text-2xl font-bold text-white">
				{#if systemInfo?.gpus?.[0]}
					{systemInfo.gpus[0].free_memory_gb?.toFixed(1)}GB
					<span class="text-sm text-gray-400">/ {systemInfo.gpus[0].total_memory_gb?.toFixed(1)}GB</span>
				{:else}
					No GPU
				{/if}
			</p>
		</div>
		
		<!-- Cache Hit Rate -->
		<div class="bg-dark-200 rounded-lg p-4">
			<div class="flex items-center gap-3 mb-2">
				<Zap size={20} class="text-yellow-400" />
				<span class="text-gray-400">Cache Hit Rate</span>
			</div>
			<p class="text-2xl font-bold text-white">
				{#if cacheStats}
					{(cacheStats.hit_rate * 100).toFixed(1)}%
				{:else}
					--
				{/if}
			</p>
		</div>
	</div>
	
	<!-- System Details -->
	<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
		<!-- System Info -->
		<div class="bg-dark-200 rounded-lg p-6">
			<h2 class="text-lg font-semibold text-white mb-4">System Information</h2>
			
			{#if systemInfo}
				<div class="space-y-3">
					<div class="flex justify-between">
						<span class="text-gray-400">Operating System</span>
						<span class="text-white">{systemInfo.os}</span>
					</div>
					<div class="flex justify-between">
						<span class="text-gray-400">CPU</span>
						<span class="text-white text-right">{systemInfo.cpu}</span>
					</div>
					<div class="flex justify-between">
						<span class="text-gray-400">CPU Cores</span>
						<span class="text-white">{systemInfo.cpu_count}</span>
					</div>
					<div class="flex justify-between">
						<span class="text-gray-400">Best Device</span>
						<span class="text-white uppercase">{systemInfo.best_device}</span>
					</div>
					
					{#if systemInfo.gpus?.length > 0}
						<div class="pt-3 border-t border-dark-100">
							<h3 class="text-sm font-medium text-gray-400 mb-2">GPU</h3>
							{#each systemInfo.gpus as gpu}
								<div class="flex justify-between">
									<span class="text-white">{gpu.name}</span>
									<span class="text-gray-400">{gpu.total_memory_gb?.toFixed(1)}GB</span>
								</div>
							{/each}
						</div>
					{/if}
				</div>
			{:else}
				<p class="text-gray-400">Loading...</p>
			{/if}
		</div>
		
		<!-- Cache Stats -->
		<div class="bg-dark-200 rounded-lg p-6">
			<div class="flex items-center justify-between mb-4">
				<h2 class="text-lg font-semibold text-white">Cache Statistics</h2>
				<button
					on:click={clearCache}
					class="text-sm text-gray-400 hover:text-white"
				>
					Clear Cache
				</button>
			</div>
			
			{#if cacheStats}
				<div class="space-y-3">
					<div class="flex justify-between">
						<span class="text-gray-400">Cache Type</span>
						<span class="text-white capitalize">{cacheStats.type}</span>
					</div>
					<div class="flex justify-between">
						<span class="text-gray-400">Entries</span>
						<span class="text-white">{cacheStats.size} / {cacheStats.max_size}</span>
					</div>
					<div class="flex justify-between">
						<span class="text-gray-400">Hits</span>
						<span class="text-green-400">{cacheStats.hits}</span>
					</div>
					<div class="flex justify-between">
						<span class="text-gray-400">Misses</span>
						<span class="text-red-400">{cacheStats.misses}</span>
					</div>
					{#if cacheStats.semantic_hits !== undefined}
						<div class="flex justify-between">
							<span class="text-gray-400">Semantic Hits</span>
							<span class="text-primary-400">{cacheStats.semantic_hits}</span>
						</div>
					{/if}
					<div class="flex justify-between">
						<span class="text-gray-400">Hit Rate</span>
						<span class="text-white">{(cacheStats.hit_rate * 100).toFixed(1)}%</span>
					</div>
				</div>
			{:else}
				<p class="text-gray-400">No cache data available</p>
			{/if}
		</div>
	</div>
</div>
