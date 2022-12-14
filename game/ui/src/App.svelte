<script>
  import Board from "./Board.svelte";
  import blackBasketImage from "./assets/black_basket.png";
  import whiteBasketImage from "./assets/white_basket.png";
  import lostImage from "./assets/youlost.png";
  import wonImage from "./assets/youwon.png";
  import { Button, Modal, ModalBody, ModalHeader } from "sveltestrap";
  import { invoke } from "@tauri-apps/api";

  let isColorSelectionOpen = true;
  let isLoseOpen = false;
  let isWonOpen = false;
  let humanPlayBlack = true;
  let blocking = false;
  let clientHeight = 10;
  let isAboutOpen = false;
  let stoneDisplay = "";

  const displayTypes = {
    "": "None",
    show_number: "Show Number",
    show_visit_times: "Show Visit Times",
  };

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
      setTimeout(() => (isWonOpen = false), 6000);
    } else if (evt.detail == "MachineWon") {
      isLoseOpen = true;
      setTimeout(() => (isLoseOpen = false), 6000);
    } else {
      // TODO : draw
    }
  };
</script>

<main class="d-flex flex-row gap-2 h-100 ">
  <div
    class="justify-content-center align-self-center h-100 position-relative"
    bind:clientHeight
    style="width: {clientHeight}px; box-shadow: 0.5rem 0 0.4rem #00000026 !important"
  >
    {#if isWonOpen || isLoseOpen}
      <div
        class="position-absolute top-0 bottom-0 start-0 end-0 d-flex align-items-center adjust-content-center justify-content-center"
        style="z-index:1000"
      >
        {#if isWonOpen}
          <img class="won_animated_text" src={wonImage} alt="You won!" />
        {:else if isLoseOpen}
          <img class="lost_animated_text" src={lostImage} alt="You lost!" />
        {/if}
      </div>
    {/if}
    <Board on:over={onGameOver} display={stoneDisplay} />
  </div>
  <div class="d-flex flex-column flex-grow-1 px-3">
    <div
      class="top_basket"
      style="background-image: url({humanPlayBlack
        ? whiteBasketImage
        : blackBasketImage})"
    />
    <div
      class="flex-grow-1 justify-content-center align-self-center d-flex align-items-center flex-column"
    >
      <div class="row">
        {#each Object.entries(displayTypes) as [value, text]}
          <div class="form-check">
            <input
              class="form-check-input"
              type="radio"
              name="display_type"
              id="display_{value}"
              checked={stoneDisplay == value}
              {value}
              on:change={() => (stoneDisplay = value)}
            />
            <label class="form-check-label" for="display_{value}">{text}</label>
          </div>
        {/each}
      </div>
      <div class="d-flex flex-row mt-3 gap-1">
        <Button
          color="secondary"
          outline
          size="sm"
          on:click={() => (isColorSelectionOpen = true)}>Restart</Button
        >
        <Button
          color="secondary"
          outline
          size="sm"
          on:click={() => (isAboutOpen = true)}>About</Button
        >
      </div>
    </div>
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

<Modal
  isOpen={isAboutOpen}
  class="modal-dialog-centered "
  toggle={() => (isAboutOpen = !isAboutOpen)}
>
  <ModalHeader>&#20851;&#20110;</ModalHeader>
  <ModalBody>
    <ul>
      <li>
        &#22522;&#20110;&#38376;&#29305;&#21345;&#27931;&#25628;&#32034;&#26641;&#19982;&#21367;&#31215;&#31070;&#32463;&#32593;&#32476;&#30340;&#24102;&#31105;&#25163;&#20116;&#23376;&#26827;&#28216;&#25103;&#65292;&#23454;&#36341;&#24378;&#21270;&#23398;&#20064;&#21450;&#20854;&#29702;&#35770;&#12290;
      </li>
      <li>
        &#20316;&#32773;&#65306; <a
          href="https://github.com/wangjia184/renju"
          class="link-dark"
          target="_blank">++</a
        >
      </li>

      <li>
        &#21442;&#32771;&#36164;&#26009;&#65306;<br />
        <ol>
          <li>
            <a
              class="link-secondary"
              href="https://joshvarty.github.io/AlphaZero/"
              target="_blank"
              >AlphaZero - A step-by-step look at Alpha Zero and Monte Carlo
              Tree Search</a
            >
          </li>
          <li>
            <a
              class="link-secondary"
              href="https://jonathan-hui.medium.com/alphago-zero-a-game-changer-14ef6e45eba5"
              target="_blank">AlphaGo Zero ??? a game changer. (How it works?)</a
            >
          </li>
          <li>
            <a
              class="link-secondary"
              href="https://jonathan-hui.medium.com/alphago-how-it-works-technically-26ddcc085319"
              target="_blank">AlphaGo: How it works technically?</a
            >
          </li>
        </ol>
      </li>
    </ul>
  </ModalBody>
</Modal>

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

  .won_animated_text {
    animation: won_frames 1s ease-in-out infinite;
  }
  @keyframes won_frames {
    0% {
      transform: scale(1);
    }
    50% {
      transform: scale(1.5);
    }
    100% {
      transform: scale(1);
    }
  }

  .lost_animated_text {
    animation: lost_frames 3s ease-in-out infinite;
  }
  @keyframes lost_frames {
    0% {
      transform: scale(1);
    }
    50% {
      transform: scale(1.2);
    }
    100% {
      transform: scale(1);
    }
  }
</style>
