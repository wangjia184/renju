#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// C API Example https://github.com/microsoft/onnxruntime-openenclave/blob/openenclave-public/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/C_Api_Sample.cpp

use std::ffi::{CStr, CString};
use std::sync::Once;
static mut API: *const OrtApi = 0 as *const OrtApi; // All onnx C API functions are defined inside OrtApi structure.
static INIT_API: Once = Once::new(); // initialize global variables only once

static mut ENV: *mut OrtEnv = 0 as *mut OrtEnv; // one enviroment per process
                                                // enviroment maintains thread pools and other state info
static INIT_ENV: Once = Once::new(); // initialize global variables only once

// retrive the API interface
// const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
fn get_api() -> *const OrtApi {
    unsafe {
        INIT_API.call_once(|| {
            let base = OrtGetApiBase();
            if let Some(get_api_fn) = (*base).GetApi {
                API = get_api_fn(ORT_API_VERSION);
            }
            assert_ne!(API, 0 as *const OrtApi);
        });
        API
    }
}

// one enviroment per process
fn get_env() -> *const OrtEnv {
    unsafe {
        INIT_ENV.call_once(|| {
            let api = OnnxApi::default();
            ENV = api.create_env("inference");
            assert_ne!(ENV, 0 as *mut OrtEnv);
        });
        ENV
    }
}

pub struct OnnxApi {
    create_env_fn: unsafe extern "C" fn(
        log_severity_level: OrtLoggingLevel,
        logid: *const ::std::os::raw::c_char,
        out: *mut *mut OrtEnv,
    ) -> OrtStatusPtr,

    set_session_graph_optimization_level_fn: unsafe extern "C" fn(
        options: *mut OrtSessionOptions,
        graph_optimization_level: GraphOptimizationLevel,
    ) -> OrtStatusPtr,

    set_intra_op_num_threads_fn: unsafe extern "C" fn(
        options: *mut OrtSessionOptions,
        intra_op_num_threads: ::std::os::raw::c_int,
    ) -> OrtStatusPtr,

    set_inter_op_num_threads_fn: unsafe extern "C" fn(
        options: *mut OrtSessionOptions,
        inter_op_num_threads: ::std::os::raw::c_int,
    ) -> OrtStatusPtr,

    create_session_options_fn:
        unsafe extern "C" fn(options: *mut *mut OrtSessionOptions) -> OrtStatusPtr,

    release_session_options_fn: unsafe extern "C" fn(input: *mut OrtSessionOptions),

    create_session_fn: unsafe extern "C" fn(
        env: *const OrtEnv,
        model_path: *const ::std::os::raw::c_char,
        options: *const OrtSessionOptions,
        out: *mut *mut OrtSession,
    ) -> OrtStatusPtr,

    release_session_fn: unsafe extern "C" fn(input: *mut OrtSession),

    get_error_message_fn:
        unsafe extern "C" fn(status: *const OrtStatus) -> *const ::std::os::raw::c_char,

    release_status: unsafe extern "C" fn(input: *mut OrtStatus),
}

impl Default for OnnxApi {
    fn default() -> Self {
        let api = unsafe { *get_api() };
        Self {
            create_env_fn: api.CreateEnv.unwrap(),
            set_intra_op_num_threads_fn: api.SetIntraOpNumThreads.unwrap(),
            set_inter_op_num_threads_fn: api.SetInterOpNumThreads.unwrap(),
            set_session_graph_optimization_level_fn: api.SetSessionGraphOptimizationLevel.unwrap(),
            create_session_options_fn: api.CreateSessionOptions.unwrap(),
            release_session_options_fn: api.ReleaseSessionOptions.unwrap(),
            create_session_fn: api.CreateSession.unwrap(),
            release_session_fn: api.ReleaseSession.unwrap(),
            get_error_message_fn: api.GetErrorMessage.unwrap(),
            release_status: api.ReleaseStatus.unwrap(),
        }
    }
}

impl OnnxApi {
    fn verify_status(self: &Self, op: &str, status_ptr: OrtStatusPtr) {
        if status_ptr != 0 as OrtStatusPtr {
            unsafe {
                let c_buf = (self.get_error_message_fn)(status_ptr);
                let error_str = CStr::from_ptr(c_buf).to_str().unwrap().to_string();
                (self.release_status)(status_ptr);
                panic!("{} : {}", op, &error_str);
            }
        }
    }
    fn create_env(self: &Self, log_id: &str) -> *mut OrtEnv {
        let mut env: *mut OrtEnv = 0 as *mut OrtEnv;
        let log_id = CString::new(log_id).expect("CString::new failed");
        let status = unsafe {
            (self.create_env_fn)(
                OrtLoggingLevel_ORT_LOGGING_LEVEL_WARNING,
                log_id.as_ptr(),
                &mut env,
            )
        };
        self.verify_status("CreateEnv", status);
        env
    }

    pub fn create_session_options(self: &Self) -> SessionOptions {
        let mut ptr: *mut OrtSessionOptions = 0 as *mut OrtSessionOptions;
        let status = unsafe { (self.create_session_options_fn)(&mut ptr) };
        self.verify_status("CreateSessionOptions", status);
        assert_ne!(ptr, 0 as *mut OrtSessionOptions);
        SessionOptions(ptr)
    }

    fn release_session_options(self: &Self, session_options: &mut SessionOptions) {
        unsafe { (self.release_session_options_fn)(session_options.0) }
        session_options.0 = 0 as *mut OrtSessionOptions;
    }

    fn set_intra_op_num_threads(self: &Self, option: &mut SessionOptions, number: i32) {
        unsafe { (self.set_intra_op_num_threads_fn)(option.0, number) };
    }

    fn set_inter_op_num_threads(self: &Self, option: &mut SessionOptions, number: i32) {
        unsafe { (self.set_inter_op_num_threads_fn)(option.0, number) };
    }

    fn set_session_graph_optimization_level(
        self: &Self,
        option: &mut SessionOptions,
        graph_optimization_level: GraphOptimizationLevel,
    ) {
        unsafe {
            (self.set_session_graph_optimization_level_fn)(option.0, graph_optimization_level)
        };
    }

    pub fn create_session(self: &Self, model_path: &str, options: &SessionOptions) -> Session {
        let mut ptr: *mut OrtSession = 0 as *mut OrtSession;
        let status_ptr = unsafe {
            let filepath = CString::new(model_path).expect("CString::new failed");
            (self.create_session_fn)(get_env(), filepath.as_ptr(), options.0, &mut ptr)
        };
        self.verify_status("CreateSession", status_ptr);
        assert_ne!(ptr, 0 as *mut OrtSession);
        Session(ptr)
    }

    fn release_session(self: &Self, session: &mut Session) {
        unsafe { (self.release_session_fn)(session.0) };
        session.0 = 0 as *mut OrtSession;
    }
}

pub struct SessionOptions(*mut OrtSessionOptions);

impl Drop for SessionOptions {
    fn drop(&mut self) {
        let api = OnnxApi::default();
        api.release_session_options(self);
    }
}

impl SessionOptions {
    pub fn set_intra_op_num_threads(self: &mut Self, number: i32) {
        let api = OnnxApi::default();
        api.set_intra_op_num_threads(self, number)
    }

    pub fn set_inter_op_num_threads(self: &mut Self, number: i32) {
        let api = OnnxApi::default();
        api.set_inter_op_num_threads(self, number)
    }

    pub fn set_session_graph_optimization_level(
        self: &mut Self,
        graph_optimization_level: GraphOptimizationLevel,
    ) {
        let api = OnnxApi::default();
        api.set_session_graph_optimization_level(self, graph_optimization_level);
    }
}

pub struct Session(*mut OrtSession);

impl Drop for Session {
    fn drop(&mut self) {
        let api = OnnxApi::default();
        api.release_session(self);
    }
}

#[test]
fn test_ort() {
    let api = OnnxApi::default();
    let mut session_options = api.create_session_options();
    session_options.set_intra_op_num_threads(1);
    session_options.set_session_graph_optimization_level(GraphOptimizationLevel_ORT_ENABLE_ALL);

    let mut session = api.create_session("model.onnx", &session_options);
}
