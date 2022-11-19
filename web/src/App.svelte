<script>
	//import wasm from "../../renjuai/Cargo.toml";
	import { onMount } from "svelte";
	/*
	onMount(async () => {
		const bindings = await wasm();
		var ret = bindings.parse("dd");
		console.log(ret);
	});
	*/
	const myWorker = new Worker("/worker.js");
</script>

<main />
