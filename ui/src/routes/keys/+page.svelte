<script>
	import { onMount } from 'svelte';
	import { Key, Plus, Copy, Trash2, Check } from 'lucide-svelte';
	
	let apiKeys = [];
	let showCreateModal = false;
	let newKeyName = '';
	let newKeyRateLimit = 100;
	let copiedKey = null;
	
	onMount(async () => {
		await loadKeys();
	});
	
	async function loadKeys() {
		try {
			const res = await fetch('/api/keys');
			if (res.ok) {
				const data = await res.json();
				apiKeys = data.keys || [];
			}
		} catch (e) {
			console.error('Failed to load API keys:', e);
		}
	}
	
	async function createKey() {
		try {
			const res = await fetch('/api/keys', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					name: newKeyName,
					rate_limit: newKeyRateLimit,
				}),
			});
			
			if (res.ok) {
				const key = await res.json();
				// Show the full key once
				alert(`Your new API key (save it now, it won't be shown again):\n\n${key.key}`);
				await loadKeys();
				showCreateModal = false;
				newKeyName = '';
			}
		} catch (e) {
			console.error('Failed to create key:', e);
		}
	}
	
	async function copyKey(key) {
		await navigator.clipboard.writeText(key);
		copiedKey = key;
		setTimeout(() => copiedKey = null, 2000);
	}
	
	async function revokeKey(key) {
		if (confirm('Are you sure you want to revoke this API key?')) {
			try {
				await fetch(`/api/keys/${key}`, { method: 'DELETE' });
				await loadKeys();
			} catch (e) {
				console.error('Failed to revoke key:', e);
			}
		}
	}
</script>

<svelte:head>
	<title>zllm - API Keys</title>
</svelte:head>

<div class="h-full p-6 overflow-y-auto">
	<div class="flex items-center justify-between mb-6">
		<h1 class="text-2xl font-bold text-white">API Keys</h1>
		<button
			on:click={() => showCreateModal = true}
			class="flex items-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-500 
						 text-white rounded-lg"
		>
			<Plus size={18} />
			<span>Create Key</span>
		</button>
	</div>
	
	<p class="text-gray-400 mb-6">
		API keys are used to authenticate requests to the zllm API. 
		Keep your keys secure and never share them publicly.
	</p>
	
	<!-- Keys List -->
	{#if apiKeys.length === 0}
		<div class="bg-dark-200 rounded-lg p-8 text-center">
			<Key size={48} class="text-gray-500 mx-auto mb-3" />
			<p class="text-gray-400">No API keys created yet.</p>
			<p class="text-gray-500 text-sm mt-1">Create a key to use the API from external applications.</p>
		</div>
	{:else}
		<div class="space-y-4">
			{#each apiKeys as key}
				<div class="bg-dark-200 rounded-lg p-4">
					<div class="flex items-center justify-between mb-2">
						<h3 class="font-medium text-white">{key.name}</h3>
						<div class="flex items-center gap-2">
							<button 
								on:click={() => copyKey(key.key)}
								class="p-2 hover:bg-dark-100 rounded-lg text-gray-400"
							>
								{#if copiedKey === key.key}
									<Check size={18} class="text-green-400" />
								{:else}
									<Copy size={18} />
								{/if}
							</button>
							<button 
								on:click={() => revokeKey(key.key)}
								class="p-2 hover:bg-dark-100 rounded-lg text-red-400"
							>
								<Trash2 size={18} />
							</button>
						</div>
					</div>
					<code class="text-sm text-gray-400 bg-dark-300 px-2 py-1 rounded">{key.key}</code>
					<div class="flex gap-4 mt-3 text-sm text-gray-500">
						<span>Created: {new Date(key.created_at).toLocaleDateString()}</span>
						<span>Requests: {key.requests}</span>
						<span class="{key.active ? 'text-green-400' : 'text-red-400'}">
							{key.active ? 'Active' : 'Revoked'}
						</span>
					</div>
				</div>
			{/each}
		</div>
	{/if}
</div>

<!-- Create Modal -->
{#if showCreateModal}
	<div class="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
		<div class="bg-dark-200 rounded-xl p-6 w-96">
			<h2 class="text-xl font-bold text-white mb-4">Create API Key</h2>
			
			<div class="space-y-4">
				<div>
					<label class="block text-sm text-gray-400 mb-1">Name</label>
					<input
						bind:value={newKeyName}
						type="text"
						placeholder="e.g., Production App"
						class="w-full bg-dark-300 border border-dark-100 rounded-lg px-3 py-2 
									 text-gray-200 outline-none focus:border-primary-500"
					/>
				</div>
				
				<div>
					<label class="block text-sm text-gray-400 mb-1">Rate Limit (requests/min)</label>
					<input
						bind:value={newKeyRateLimit}
						type="number"
						class="w-full bg-dark-300 border border-dark-100 rounded-lg px-3 py-2 
									 text-gray-200 outline-none focus:border-primary-500"
					/>
				</div>
			</div>
			
			<div class="flex justify-end gap-3 mt-6">
				<button
					on:click={() => showCreateModal = false}
					class="px-4 py-2 text-gray-400 hover:text-gray-200"
				>
					Cancel
				</button>
				<button
					on:click={createKey}
					disabled={!newKeyName.trim()}
					class="px-4 py-2 bg-primary-600 hover:bg-primary-500 disabled:bg-gray-600 
								 text-white rounded-lg"
				>
					Create Key
				</button>
			</div>
		</div>
	</div>
{/if}
