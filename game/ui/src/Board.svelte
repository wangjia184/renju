<script>
    import boardImage from "./assets/board.jpeg";

    let width, height;
</script>

<div
    class="squared_board"
    style="background-image: url({boardImage});"
    bind:clientWidth={width}
    bind:clientHeight={height}
>
    <div class="stone" />
</div>

<style>
    .squared_board {
        aspect-ratio: 1 / 1; /* ⏹ a perfect square */
        min-width: 800px;
        min-height: 800px;
        max-width: 1000px;
        max-height: 1000px;
        background-repeat: no-repeat;
        background-size: 100% auto;
        position: relative;
    }

    .stone {
        position: absolute;
        width: 100px;
        height: 100px;
        background-color: rgba(0, 0, 0, 0.5);
    }
</style>
