<script>
    import boardImage from "./assets/bg.jpeg";
    import blackImage from "./assets/black.png";
    import whiteImage from "./assets/white.png";
    import { Canvas, Layer, t } from "svelte-canvas";
    import { invoke } from "@tauri-apps/api";
    import { listen } from "@tauri-apps/api/event";
    import { onMount } from "svelte";

    import { createEventDispatcher } from "svelte";
    const dispatch = createEventDispatcher();

    const MARGIN = 50;
    const SIZE = 15.0;

    export let display = "";

    let container;
    let boardWidth, boardHeight;
    let stoneWidth, stoneHeight;
    let stoneMatrix = [];
    let blocking = false;
    let state = "";
    let lastMove = null;
    $: stoneWidth = (boardWidth - 2 * MARGIN) / (SIZE - 1);
    $: stoneHeight = (boardHeight - 2 * MARGIN) / (SIZE - 1);

    onMount(async () => {
        const unlisten = await listen("board_updated", (evt) => {
            if (evt.payload) {
                stoneMatrix = evt.payload.matrix;
                state = evt.payload.state;
                lastMove = evt.payload.last;
            }
        });
        return unlisten;
    });

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

        if (!blocking && state == "HumanThinking") {
            blocking = true;
            invoke("do_move", { pos: [row, col] }).then((state) => {
                if (
                    state == "MachineWon" ||
                    state == "HumanWon" ||
                    state == "Draw"
                ) {
                    dispatch("over", state);
                }
                blocking = false;
            });
        }
    };

    const getStoneImage = (value) => {
        return value > 0 ? (value % 2 == 0 ? whiteImage : blackImage) : "";
    };
</script>

<div
    style="background-image: url({boardImage});"
    class="squared_board {display}"
    bind:this={container}
    bind:clientWidth={boardWidth}
    bind:clientHeight={boardHeight}
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
    :global(.show_number) .stone_number {
        display: inline-block !important;
    }

    .black_stone .stone_number {
        color: white;
    }
    .white_stone .stone_number {
        color: black;
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
