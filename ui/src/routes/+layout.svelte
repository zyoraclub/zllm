<script>
	import '../app.css';
	import Sidebar from '$lib/components/Sidebar.svelte';
	import Header from '$lib/components/Header.svelte';
</script>

<div class="flex h-screen bg-dark-400">
	<Sidebar />
	<div class="flex-1 flex flex-col">
		<Header />
		<main class="flex-1 overflow-hidden">
			<slot />
		</main>
	</div>
</div>
