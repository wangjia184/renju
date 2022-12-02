<script>
    import { Canvas, Layer, t } from "svelte-canvas";
    import { onMount } from "svelte";

    import { createEventDispatcher } from "svelte";
    const dispatch = createEventDispatcher();

    const MARGIN = 50;
    const SIZE = 15.0;

    export let display = "show_probability";

    export let boardstate = null;

    $: boardstate,
        (() => {
            if (boardstate) {
                console.log(boardstate);
                stoneMatrix = boardstate.matrix;
                visitCountMatrix = boardstate.visit_count;
                probabilityMatrix = boardstate.probability;
                avgValueMatrix = boardstate.avg_value;
                const state = boardstate.state;
                lastMove = boardstate.last;

                if (
                    state == "MachineWon" ||
                    state == "HumanWon" ||
                    state == "Draw"
                ) {
                    dispatch("over", state);
                }
            }
        })();

    let container;
    let boardWidth, boardHeight;
    let stoneWidth, stoneHeight;
    let stoneMatrix = [];
    let visitCountMatrix = [];
    let probabilityMatrix = [];
    let avgValueMatrix = [];

    let lastMove = null;
    $: stoneWidth = (boardWidth - 2 * MARGIN) / (SIZE - 1);
    $: stoneHeight = (boardHeight - 2 * MARGIN) / (SIZE - 1);

    onMount(async () => {});

    $: renderGrid = ({ context, width, height }) => {
        context.beginPath();
        context.lineWidth = 5;
        context.strokeStyle = "rgb(0, 0, 0, 0.8)";
        context.rect(MARGIN, MARGIN, width - MARGIN * 2, height - MARGIN * 2);
        context.stroke(); // draw it
        context.closePath();

        for (var i = 1; i < SIZE; i++) {
            // vertical line
            context.beginPath();
            context.lineWidth = 2;
            context.strokeStyle = "rgb(0, 0, 0, 0.8)";
            context.moveTo(MARGIN + i * stoneWidth, MARGIN);
            context.lineTo(MARGIN + i * stoneWidth, height - MARGIN);
            context.stroke(); // draw it
            context.closePath();

            // Horizontal line
            context.beginPath();
            context.lineWidth = 2;
            context.strokeStyle = "rgb(0, 0, 0, 0.8)";
            context.moveTo(MARGIN, MARGIN + i * stoneHeight);
            context.lineTo(width - MARGIN, MARGIN + i * stoneHeight);
            context.stroke(); // draw it
            context.closePath();
        }

        const points = [
            [3, 3],
            [7, 3],
            [11, 3],
            [3, 7],
            [7, 7],
            [11, 7],
            [3, 11],
            [7, 11],
            [11, 11],
        ];
        points.forEach((pos) => {
            const x = MARGIN + pos[0] * stoneWidth;
            const y = MARGIN + pos[1] * stoneHeight;

            context.beginPath();
            context.fillStyle = "rgb(0, 0, 0, 0.8)";
            context.arc(x, y, 6, 0, 2 * Math.PI);
            context.fill();
            context.closePath();
        });
    };

    const onPositionClicked = (evt) => {
        const rc = container.getBoundingClientRect();
        const horizontal =
            (evt.clientX - rc.x - (MARGIN - stoneWidth)) / stoneWidth;
        const vertical =
            (evt.clientY - rc.y - (MARGIN - stoneHeight)) / stoneHeight;

        let row, col;
        if (vertical % 1 < 0.4) {
            row = Math.floor(vertical) - 1;
        } else if (vertical % 1 > 0.6) {
            row = Math.ceil(vertical) - 1;
        } else {
            return;
        }
        if (horizontal % 1 < 0.4) {
            col = Math.floor(horizontal) - 1;
        } else if (horizontal % 1 > 0.6) {
            col = Math.ceil(horizontal) - 1;
        } else {
            return;
        }

        dispatch("stone", { pos: [row, col] });
    };

    const getStoneImage = (value) => {
        return value > 0
            ? value % 2 == 0
                ? "/assets/white.png"
                : "/assets/black.png"
            : "";
    };
</script>

<div
    class="squared_board {display}"
    bind:this={container}
    bind:clientWidth={boardWidth}
    bind:clientHeight={boardHeight}
    on:keydown={() => {}}
    on:click={onPositionClicked}
>
    <Canvas width={boardWidth} height={boardHeight}>
        <Layer render={renderGrid} />
    </Canvas>

    {#each stoneMatrix as rowArray, row}
        {#each rowArray as value, col}
            <div
                class="stone d-flex justify-content-center align-items-center
                    {value % 2 > 0 && 'black_stone'}
                    {value % 2 == 0 && value && 'white_stone'}
                    {lastMove &&
                    lastMove.length == 2 &&
                    lastMove[0] == row &&
                    lastMove[1] == col &&
                    'last-move'}"
                style="background-image : url({getStoneImage(value)});
                left:{MARGIN + (col - 0.5) * stoneWidth + 2}px;
                top:{MARGIN + (row - 0.5) * stoneHeight + 2}px; 
                width:{stoneWidth - 4}px; 
                height:{stoneHeight - 4}px"
            >
                <span class="stone_number">{value ? value : ""}</span>

                {#if visitCountMatrix[row][col] > 0}
                    <span class="visit_count">
                        {#if visitCountMatrix[row][col] > 10000000}
                            {(visitCountMatrix[row][col] / 1000000).toFixed(0)} M
                        {:else if visitCountMatrix[row][col] > 1000000}
                            {(visitCountMatrix[row][col] / 1000000).toFixed(1)} M
                        {:else if visitCountMatrix[row][col] > 10000}
                            {(visitCountMatrix[row][col] / 1000).toFixed(0)} K
                        {:else if visitCountMatrix[row][col] > 1000}
                            {(visitCountMatrix[row][col] / 1000).toFixed(1)} K
                        {:else}
                            {visitCountMatrix[row][col]}
                        {/if}
                    </span>
                {/if}

                {#if probabilityMatrix[row][col] > 0.0001}
                    <span class="probability">
                        {#if probabilityMatrix[row][col] > 0.01}
                            {(probabilityMatrix[row][col] * 100).toFixed(0)}
                        {:else}
                            {(probabilityMatrix[row][col] * 100).toFixed(2)}
                        {/if}
                        %
                    </span>
                {/if}

                {#if Math.abs(avgValueMatrix[row][col]) > 0.01}
                    <span class="avg_value">
                        {#if Math.abs(avgValueMatrix[row][col]) > 0.01}
                            {(avgValueMatrix[row][col] * 100).toFixed(0)}
                        {:else}
                            {(avgValueMatrix[row][col] * 100).toFixed(2)}
                        {/if}
                    </span>
                {/if}
            </div>
        {/each}
    {/each}
</div>

<style>
    .squared_board {
        aspect-ratio: 1 / 1; /* ⏹ a perfect square */
        height: 100%;
        background-repeat: no-repeat;
        background-size: 100% auto;
        position: relative;
        margin: 0 auto;
        background-image: url(/assets/bg.jpeg);
    }

    .stone {
        position: absolute;
        background-position: center;
        background-repeat: no-repeat;
        background-size: 85%;
        cursor: default;
    }
    .last-move {
        border-radius: 25px;
        animation: blink 3s infinite;
    }
    .stone_number {
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        font-size: 16px;
        cursor: default;
        display: none;
    }
    .visit_count {
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        font-size: 12px;
        cursor: default;
        color: #66ff66;
        text-shadow: 1px 1px 1px black, -1px -1px 1px black, -1px 1px 1px black,
            1px -1px 1px black;
        display: none;
    }
    .probability {
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        font-size: 12px;
        cursor: default;
        color: #ee6666;
        text-shadow: 1px 1px 1px black, -1px -1px 1px black, -1px 1px 1px black,
            1px -1px 1px black;
        display: none;
    }
    .avg_value {
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        font-size: 12px;
        cursor: default;
        color: yellow;
        text-shadow: 1px 1px 1px black, -1px -1px 1px black, -1px 1px 1px black,
            1px -1px 1px black;
        display: none;
    }
    :global(.show_number) .stone_number {
        display: inline-block !important;
    }
    :global(.show_visit_count) .visit_count {
        display: inline-block !important;
    }
    :global(.show_probability) .probability {
        display: inline-block !important;
    }
    :global(.show_avg_value) .avg_value {
        display: inline-block !important;
    }

    .black_stone .stone_number {
        color: white;
    }
    .white_stone .stone_number {
        color: black;
    }
    .black_stone .visit_count {
        text-shadow: none;
    }

    @keyframes blink {
        0% {
            background-color: rgb(255, 255, 255, 0);
        }
        50% {
            background-color: rgb(255, 255, 255, 0.5);
        }
        100% {
            background-color: rgb(255, 255, 255, 0);
        }
    }
</style>
