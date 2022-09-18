<script>
    import boardImage from "./assets/bg.jpeg";
    import blackImage from "./assets/black.png";
    import whiteImage from "./assets/white.png";
    import { Canvas, Layer, t } from "svelte-canvas";
    import { invoke } from "@tauri-apps/api";
    import { emit, listen } from "@tauri-apps/api/event";
    import { onMount } from "svelte";

    import { createEventDispatcher } from "svelte";
    const dispatch = createEventDispatcher();

    const MARGIN = 50;
    const SIZE = 15.0;
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

    const getStoneImage = (color) => {
        if (color == 1) {
            return blackImage;
        }
        if (color == 2) {
            return whiteImage;
        }
        return "";
    };

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
        const horizontal = (evt.clientX - rc.x - (MARGIN - stoneWidth)) / stoneWidth;
        const vertical = (evt.clientY -rc.y - (MARGIN - stoneHeight)) / stoneHeight;

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
</script>

<div
    class="squared_board"
    style="background-image: url({boardImage});"
    bind:this={container}
    bind:clientWidth={boardWidth}
    bind:clientHeight={boardHeight}
    on:click={onPositionClicked}
>
    <Canvas width={boardWidth} height={boardHeight}>
        <Layer render={renderGrid} />
    </Canvas>

    {#each stoneMatrix as rowArray, row}
        {#each rowArray as color, col}
            <div
                class="stone {lastMove &&
                    lastMove.length == 2 &&
                    lastMove[0] == row &&
                    lastMove[1] == col &&
                    'last-move'}"
                style="background-image: url({getStoneImage(color)});
                left:{MARGIN + (col - 0.5) * stoneWidth + 2}px; 
                top:{MARGIN + (row - 0.5) * stoneHeight + 2}px; 
                width:{stoneWidth - 4}px; 
                height:{stoneHeight - 4}px"
            />
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
        display: block;
        background-position: center;
        background-repeat: no-repeat;
        background-size: 85%;
        cursor: default;
    }
    .last-move {
        border-radius: 25px;
        animation: mymove 3s infinite;
    }

    @keyframes mymove {
        from {
            background-color: rgb(255, 255, 255, 0.3);
        }
        to {
            background-color: rgb(255, 255, 255, 0);
        }
    }
</style>
