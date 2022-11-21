/* tslint:disable */
/* eslint-disable */
/**
* @param {string} input
* @returns {Promise<any>}
*/
export function test(input: string): Promise<any>;
/**
*/
export enum MatchState {
  HumanThinking,
  MachineThinking,
  Draw,
  HumanWon,
  MachineWon,
}
/**
*/
export class BoardInfo {
  free(): void;
}
/**
*/
export class Brain {
  free(): void;
/**
*/
  constructor();
/**
* @param {boolean} human_play_black
* @returns {any}
*/
  reset(human_play_black: boolean): any;
/**
* @param {number} row
* @param {number} col
* @returns {Promise<any>}
*/
  human_move(row: number, col: number): Promise<any>;
/**
* @param {number} iterations
* @returns {Promise<void>}
*/
  think(iterations: number): Promise<void>;
/**
* @returns {Promise<any>}
*/
  machine_move(): Promise<any>;
}
/**
*/
export class Prediction {
  free(): void;
/**
*/
  constructor();
/**
*/
  probabilities: Array<any>;
/**
*/
  score: number;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_prediction_free: (a: number) => void;
  readonly prediction_new: () => number;
  readonly prediction_set_probabilities: (a: number, b: number) => void;
  readonly prediction_set_score: (a: number, b: number) => void;
  readonly test: (a: number, b: number) => number;
  readonly __wbg_boardinfo_free: (a: number) => void;
  readonly __wbg_brain_free: (a: number) => void;
  readonly brain_new: () => number;
  readonly brain_reset: (a: number, b: number) => number;
  readonly brain_human_move: (a: number, b: number, c: number) => number;
  readonly brain_think: (a: number, b: number) => number;
  readonly brain_machine_move: (a: number) => number;
  readonly __wbindgen_malloc: (a: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number) => number;
  readonly __wbindgen_export_2: WebAssembly.Table;
  readonly _dyn_core__ops__function__FnMut__A____Output___R_as_wasm_bindgen__closure__WasmClosure___describe__invoke__hae7b63785a4161f1: (a: number, b: number, c: number) => void;
  readonly __wbindgen_free: (a: number, b: number) => void;
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly wasm_bindgen__convert__closures__invoke2_mut__h757675fbe177217d: (a: number, b: number, c: number, d: number) => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {SyncInitInput} module
*
* @returns {InitOutput}
*/
export function initSync(module: SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {InitInput | Promise<InitInput>} module_or_path
*
* @returns {Promise<InitOutput>}
*/
export default function init (module_or_path?: InitInput | Promise<InitInput>): Promise<InitOutput>;
