/* tslint:disable */
/* eslint-disable */
/**
* @param {string} s
* @returns {Car}
*/
export function greet(s: string): Car;
/**
* @param {string} input
* @returns {Promise<any>}
*/
export function test(input: string): Promise<any>;
/**
* @param {boolean} human_play_black
*/
export function start(human_play_black: boolean): void;
/**
*/
export class Car {
  free(): void;
/**
*/
  color: number;
/**
*/
  number: number;
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
  readonly greet: (a: number, b: number) => number;
  readonly __wbg_car_free: (a: number) => void;
  readonly __wbg_get_car_number: (a: number) => number;
  readonly __wbg_set_car_number: (a: number, b: number) => void;
  readonly __wbg_get_car_color: (a: number) => number;
  readonly __wbg_set_car_color: (a: number, b: number) => void;
  readonly __wbg_prediction_free: (a: number) => void;
  readonly prediction_new: () => number;
  readonly prediction_set_probabilities: (a: number, b: number) => void;
  readonly prediction_set_score: (a: number, b: number) => void;
  readonly test: (a: number, b: number) => number;
  readonly start: (a: number) => void;
  readonly __wbindgen_malloc: (a: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number) => number;
  readonly __wbindgen_export_2: WebAssembly.Table;
  readonly _dyn_core__ops__function__FnMut__A____Output___R_as_wasm_bindgen__closure__WasmClosure___describe__invoke__h5d8269c182510fce: (a: number, b: number, c: number) => void;
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly wasm_bindgen__convert__closures__invoke2_mut__h9742c000f3195eb0: (a: number, b: number, c: number, d: number) => void;
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
