<script>
  import Board from "./Board.svelte";
  import blackBasketImage from "./assets/black_basket.png";
  import whiteBasketImage from "./assets/white_basket.png";
  import lostImage from "./assets/lost.gif";
  import wonImage from "./assets/won.gif";
  import { Modal, ModalBody, ModalHeader, Offcanvas } from "sveltestrap";
  import { invoke } from "@tauri-apps/api";

  let isColorSelectionOpen = true;
  let isLoseOpen = false;
  let isWonOpen = false;
  let humanPlayBlack = true;
  let blocking = false;
  const startNewGame = (black) => {
    if (blocking) {
      return;
    }
    blocking = true;
    humanPlayBlack = black;
    invoke("new_match", { black: black }).then((response) => {
      isColorSelectionOpen = false;
      blocking = false;
    });
  };

  const onGameOver = (evt) => {
    if (evt.detail == "HumanWon") {
      isWonOpen = true;
    } else if (evt.detail == "MachineWon") {
      isLoseOpen = true;
    } else {
      // TODO : draw
    }
  };
</script>

<main class="d-flex flex-row gap-2 h-100">
  <div class="justify-content-center align-self-center">
    <Board on:over={onGameOver} />
  </div>
  <div class="d-flex flex-column flex-grow-1">
    <div
      class="top_basket"
      style="background-image: url({humanPlayBlack
        ? whiteBasketImage
        : blackBasketImage})"
    />
    <div class="flex-grow-1" />
    <div
      class="bottom_basket"
      style="background-image: url({humanPlayBlack
        ? blackBasketImage
        : whiteBasketImage})"
    />
  </div>
</main>

<Modal isOpen={isColorSelectionOpen} size="xl" class="modal-dialog-centered ">
  <ModalHeader>Choose your stone color</ModalHeader>
  <ModalBody>
    <div class="d-flex flex-row mb-4">
      <div class="flex-grow-1 text-center">
        <img
          src={blackBasketImage}
          alt="Black"
          class="color_selector"
          on:click={() => startNewGame(true)}
        />
      </div>
      <div class="flex-grow-1 text-center">
        <img
          src={whiteBasketImage}
          alt="White"
          class="color_selector"
          on:click={() => startNewGame(false)}
        />
      </div>
    </div>
  </ModalBody>
</Modal>

<Offcanvas
  isOpen={isWonOpen}
  placement="top"
  header="Congratulations! You Won!"
  toggle={() => (isWonOpen = !isWonOpen)}
>
  <div class="w-100 h-100 won_lost" style="background-image:url({wonImage})" />
</Offcanvas>

<Offcanvas
  isOpen={isLoseOpen}
  placement="top"
  header="Oh No! You Lost!"
  toggle={() => (isLoseOpen = !isLoseOpen)}
>
  <div class="w-100 h-100 won_lost" style="background-image:url({lostImage})" />
</Offcanvas>

<style>
  .top_basket {
    width: 100%;
    height: 200px;
    background-repeat: no-repeat;
    background-size: contain;
    background-position: center top;
  }

  .bottom_basket {
    width: 100%;
    height: 200px;
    background-repeat: no-repeat;
    background-size: contain;
    background-position: center bottom;
  }

  .color_selector {
    cursor: pointer;
    opacity: 0.5;
  }

  .color_selector:hover {
    opacity: 1;
  }

  .won_lost {
    background-size: 100%;
    background-repeat: no-repeat;
    background-position: center;
  }
</style>
