#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// API doc : https://onnxruntime.ai/docs/api/c/struct_ort_api.html
// C API Example https://github.com/microsoft/onnxruntime-openenclave/blob/openenclave-public/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/C_Api_Sample.cpp

use crate::game::{SquareMatrix, StateTensor, StateTensorExtension};
use std::any::TypeId;
use std::ffi::{CStr, CString};
use std::mem;
use std::os::raw::{c_char, c_void};
use std::ptr;
use std::sync::Once;

lazy_static! {
    static ref API: OnnxApi = unsafe {
        let base = OrtGetApiBase();
        assert_ne!(base, 0 as *const OrtApiBase);
        if let Some(get_api_fn) = (*base).GetApi {
            let ptr = get_api_fn(ORT_API_VERSION);
            assert_ne!(ptr, 0 as *const OrtApi);
            OnnxApi::new(ptr)
        } else {
            panic!("GetApi is missing");
        }
    };
}

static mut ENV: *mut OrtEnv = 0 as *mut OrtEnv; // one enviroment per process
                                                // enviroment maintains thread pools and other state info
static INIT_ENV: Once = Once::new(); // initialize global variables only once

// one enviroment per process
fn get_env() -> *const OrtEnv {
    unsafe {
        INIT_ENV.call_once(|| {
            ENV = API.create_env("inference");
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

    session_get_input_count_fn:
        unsafe extern "C" fn(session: *const OrtSession, out: *mut usize) -> OrtStatusPtr,

    session_get_output_count_fn:
        unsafe extern "C" fn(session: *const OrtSession, out: *mut usize) -> OrtStatusPtr,

    session_get_input_name_fn: unsafe extern "C" fn(
        session: *const OrtSession,
        index: usize,
        allocator: *mut OrtAllocator,
        value: *mut *mut ::std::os::raw::c_char,
    ) -> OrtStatusPtr,

    session_get_output_name_fn: unsafe extern "C" fn(
        session: *const OrtSession,
        index: usize,
        allocator: *mut OrtAllocator,
        value: *mut *mut ::std::os::raw::c_char,
    ) -> OrtStatusPtr,

    session_get_input_type_info_fn: unsafe extern "C" fn(
        session: *const OrtSession,
        index: usize,
        type_info: *mut *mut OrtTypeInfo,
    ) -> OrtStatusPtr,

    session_get_output_type_info_fn: unsafe extern "C" fn(
        session: *const OrtSession,
        index: usize,
        type_info: *mut *mut OrtTypeInfo,
    ) -> OrtStatusPtr,

    get_allocator_with_default_options_fn:
        unsafe extern "C" fn(out: *mut *mut OrtAllocator) -> OrtStatusPtr,

    get_error_message_fn:
        unsafe extern "C" fn(status: *const OrtStatus) -> *const ::std::os::raw::c_char,

    release_status: unsafe extern "C" fn(input: *mut OrtStatus),

    create_cpu_memory_info_fn: unsafe extern "C" fn(
        type_: OrtAllocatorType,
        mem_type: OrtMemType,
        out: *mut *mut OrtMemoryInfo,
    ) -> OrtStatusPtr,

    create_tensor_with_data_as_ort_value_fn: unsafe extern "C" fn(
        info: *const OrtMemoryInfo,
        p_data: *mut ::std::os::raw::c_void,
        p_data_len: usize,
        shape: *const i64,
        shape_len: usize,
        data_type: ONNXTensorElementDataType,
        out: *mut *mut OrtValue,
    ) -> OrtStatusPtr,

    release_memory_info_fn: unsafe extern "C" fn(input: *mut OrtMemoryInfo),

    release_value_fn: unsafe extern "C" fn(input: *mut OrtValue),

    run_fn: unsafe extern "C" fn(
        session: *mut OrtSession,
        run_options: *const OrtRunOptions,
        input_names: *const *const ::std::os::raw::c_char,
        inputs: *const *const OrtValue,
        input_len: usize,
        output_names: *const *const ::std::os::raw::c_char,
        output_names_len: usize,
        outputs: *mut *mut OrtValue,
    ) -> OrtStatusPtr,

    get_tensor_mutable_data_fn: unsafe extern "C" fn(
        value: *mut OrtValue,
        out: *mut *mut ::std::os::raw::c_void,
    ) -> OrtStatusPtr,

    get_tensor_type_and_shape: unsafe extern "C" fn(
        value: *const OrtValue,
        out: *mut *mut OrtTensorTypeAndShapeInfo,
    ) -> OrtStatusPtr,

    cast_type_info_to_tensor_info_fn: unsafe extern "C" fn(
        type_info: *const OrtTypeInfo,
        out: *mut *const OrtTensorTypeAndShapeInfo,
    ) -> OrtStatusPtr,

    get_tensor_element_type_fn: unsafe extern "C" fn(
        info: *const OrtTensorTypeAndShapeInfo,
        out: *mut ONNXTensorElementDataType,
    ) -> OrtStatusPtr,

    get_dimensions_count_fn: unsafe extern "C" fn(
        info: *const OrtTensorTypeAndShapeInfo,
        out: *mut usize,
    ) -> OrtStatusPtr,

    get_dimensions_fn: unsafe extern "C" fn(
        info: *const OrtTensorTypeAndShapeInfo,
        dim_values: *mut i64,
        dim_values_length: usize,
    ) -> OrtStatusPtr,

    release_type_info_fn: unsafe extern "C" fn(input: *mut OrtTypeInfo),
}

impl OnnxApi {
    fn new(ptr: *const OrtApi) -> Self {
        let api = unsafe { *ptr };
        Self {
            create_env_fn: api.CreateEnv.unwrap(),
            set_intra_op_num_threads_fn: api.SetIntraOpNumThreads.unwrap(),
            set_inter_op_num_threads_fn: api.SetInterOpNumThreads.unwrap(),
            set_session_graph_optimization_level_fn: api.SetSessionGraphOptimizationLevel.unwrap(),
            create_session_options_fn: api.CreateSessionOptions.unwrap(),
            release_session_options_fn: api.ReleaseSessionOptions.unwrap(),
            create_session_fn: api.CreateSession.unwrap(),
            release_session_fn: api.ReleaseSession.unwrap(),
            session_get_input_count_fn: api.SessionGetInputCount.unwrap(),
            session_get_output_count_fn: api.SessionGetOutputCount.unwrap(),
            session_get_input_name_fn: api.SessionGetInputName.unwrap(),
            session_get_output_name_fn: api.SessionGetOutputName.unwrap(),
            session_get_input_type_info_fn: api.SessionGetInputTypeInfo.unwrap(),
            session_get_output_type_info_fn: api.SessionGetOutputTypeInfo.unwrap(),
            get_allocator_with_default_options_fn: api.GetAllocatorWithDefaultOptions.unwrap(),
            get_error_message_fn: api.GetErrorMessage.unwrap(),
            release_status: api.ReleaseStatus.unwrap(),
            create_cpu_memory_info_fn: api.CreateCpuMemoryInfo.unwrap(),
            create_tensor_with_data_as_ort_value_fn: api.CreateTensorWithDataAsOrtValue.unwrap(),
            release_memory_info_fn: api.ReleaseMemoryInfo.unwrap(),
            release_value_fn: api.ReleaseValue.unwrap(),
            run_fn: api.Run.unwrap(),
            get_tensor_mutable_data_fn: api.GetTensorMutableData.unwrap(),
            get_tensor_type_and_shape: api.GetTensorTypeAndShape.unwrap(),
            cast_type_info_to_tensor_info_fn: api.CastTypeInfoToTensorInfo.unwrap(),
            get_tensor_element_type_fn: api.GetTensorElementType.unwrap(),
            get_dimensions_count_fn: api.GetDimensionsCount.unwrap(),
            get_dimensions_fn: api.GetDimensions.unwrap(),
            release_type_info_fn: api.ReleaseTypeInfo.unwrap(),
        }
    }

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

    pub fn create_session(self: &Self, model_path: &str, options: &SessionOptions) -> Session {
        let mut ptr: *mut OrtSession = 0 as *mut OrtSession;
        let status_ptr = unsafe {
            let filepath = CString::new(model_path).expect("CString::new failed");
            (self.create_session_fn)(get_env(), filepath.as_ptr(), options.0, &mut ptr)
        };
        self.verify_status("CreateSession", status_ptr);
        assert_ne!(ptr, 0 as *mut OrtSession);

        let mut allocator_ptr: *mut OrtAllocator = 0 as *mut OrtAllocator;
        let status_ptr =
            unsafe { (self.get_allocator_with_default_options_fn)(&mut allocator_ptr) };
        self.verify_status("GetAllocatorWithDefaultOptions", status_ptr);
        assert_ne!(allocator_ptr, 0 as *mut OrtAllocator);

        let mut input_count: usize = 0;
        let status_ptr = unsafe { (self.session_get_input_count_fn)(ptr, &mut input_count) };
        self.verify_status("SessionGetInputCount", status_ptr);
        assert_ne!(input_count, 0);

        let mut output_count: usize = 0;
        let status_ptr = unsafe { (self.session_get_output_count_fn)(ptr, &mut output_count) };
        self.verify_status("SessionGetOutputCount", status_ptr);
        assert_ne!(output_count, 0);

        let mut input_tensors: Vec<TensorInfo> = Vec::with_capacity(input_count);
        let mut input_names: Vec<*const c_char> = Vec::with_capacity(input_count);
        for index in 0..input_count {
            let mut name: *mut c_char = 0 as *mut c_char;
            let status_ptr =
                unsafe { (self.session_get_input_name_fn)(ptr, index, allocator_ptr, &mut name) };
            self.verify_status("SessionGetInputName", status_ptr);
            input_names.push(name);

            let mut type_info_ptr = 0 as *mut OrtTypeInfo;
            let status_ptr =
                unsafe { (self.session_get_input_type_info_fn)(ptr, index, &mut type_info_ptr) };
            self.verify_status("SessionGetInputTypeInfo", status_ptr);

            let type_info = TypeInfo(type_info_ptr);
            let type_and_shape = type_info.get_type_and_shape();

            input_tensors.push(TensorInfo {
                element_type: type_and_shape.0,
                shape: type_and_shape.1,
                name: unsafe { CStr::from_ptr(name).to_str().unwrap().to_string() },
            })
        }

        assert_eq!(input_tensors.len(), 1);
        assert_eq!(
            input_tensors[0].element_type,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        );

        let mut output_tensors: Vec<TensorInfo> = Vec::with_capacity(output_count);
        let mut output_names: Vec<*const c_char> = Vec::with_capacity(output_count);
        for index in 0..output_count {
            let mut name: *mut c_char = 0 as *mut c_char;
            let status_ptr =
                unsafe { (self.session_get_output_name_fn)(ptr, index, allocator_ptr, &mut name) };
            self.verify_status("SessionGetOutputName", status_ptr);
            output_names.push(name);

            let mut type_info_ptr = 0 as *mut OrtTypeInfo;
            let status_ptr =
                unsafe { (self.session_get_output_type_info_fn)(ptr, index, &mut type_info_ptr) };
            self.verify_status("SessionGetOutputTypeInfo", status_ptr);

            let type_info = TypeInfo(type_info_ptr);
            let type_and_shape = type_info.get_type_and_shape();

            output_tensors.push(TensorInfo {
                element_type: type_and_shape.0,
                shape: type_and_shape.1,
                name: unsafe { CStr::from_ptr(name).to_str().unwrap().to_string() },
            })
        }

        assert_eq!(output_tensors.len(), 2);
        assert_eq!(
            output_tensors[0].element_type,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        );
        assert_eq!(
            output_tensors[1].element_type,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        );

        Session {
            ptr: ptr,
            input_names: input_names,
            output_names: output_names,
            input_tensors: input_tensors,
            output_tensors: output_tensors,
        }
    }

    pub fn create_cpu_memory_info(
        self: &Self,
        allocator_type: OrtAllocatorType,
        mem_type: OrtMemType,
    ) -> MemoryInfo {
        let mut ptr: *mut OrtMemoryInfo = 0 as *mut OrtMemoryInfo;
        let status_ptr =
            unsafe { (self.create_cpu_memory_info_fn)(allocator_type, mem_type, &mut ptr) };
        self.verify_status("CreateCpuMemoryInfo", status_ptr);
        assert_ne!(ptr, 0 as *mut OrtMemoryInfo);
        MemoryInfo(ptr)
    }

    fn create_tensor_with_data_as_ort_value(
        self: &Self,
        memory_info: &MemoryInfo,
        p_data: *mut c_void,
        p_data_len: usize,
        shape: *const i64,
        shape_len: usize,
        data_type: ONNXTensorElementDataType,
    ) -> Tensor {
        let mut ptr: *mut OrtValue = 0 as *mut OrtValue;
        let status_ptr = unsafe {
            (self.create_tensor_with_data_as_ort_value_fn)(
                memory_info.0,
                p_data,
                p_data_len,
                shape,
                shape_len,
                data_type,
                &mut ptr,
            )
        };
        self.verify_status("CreateTensorWithDataAsOrtValue", status_ptr);
        assert_ne!(ptr, 0 as *mut OrtValue);
        Tensor::new(ptr)
    }
}

pub struct SessionOptions(*mut OrtSessionOptions);

impl Drop for SessionOptions {
    fn drop(&mut self) {
        unsafe { (API.release_session_options_fn)(self.0) }
        self.0 = 0 as *mut OrtSessionOptions;
    }
}

impl SessionOptions {
    pub fn set_intra_op_num_threads(self: &mut Self, number: i32) {
        unsafe { (API.set_intra_op_num_threads_fn)(self.0, number) };
    }

    pub fn set_inter_op_num_threads(self: &mut Self, number: i32) {
        unsafe { (API.set_inter_op_num_threads_fn)(self.0, number) };
    }

    pub fn set_session_graph_optimization_level(
        self: &mut Self,
        graph_optimization_level: GraphOptimizationLevel,
    ) {
        unsafe { (API.set_session_graph_optimization_level_fn)(self.0, graph_optimization_level) };
    }
}

#[derive(Debug)]
pub struct TensorInfo {
    element_type: ONNXTensorElementDataType,
    shape: Vec<i64>,
    name: String,
}

#[derive(Debug)]
pub struct Session {
    ptr: *mut OrtSession,
    input_names: Vec<*const c_char>,
    output_names: Vec<*const c_char>,
    input_tensors: Vec<TensorInfo>,
    output_tensors: Vec<TensorInfo>,
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe { (API.release_session_fn)(self.ptr) };
        self.ptr = 0 as *mut OrtSession;
    }
}

impl Session {
    pub fn run(&self, input_tensors: &Tensor) -> Vec<Tensor> {
        let mut output_values: Vec<*mut OrtValue> =
            vec![0 as *mut OrtValue; self.output_names.len()];
        let inputs = vec![input_tensors.ptr as *const OrtValue];

        let status_ptr = unsafe {
            (API.run_fn)(
                self.ptr,
                0 as *const OrtRunOptions,
                self.input_names.as_ptr(),
                inputs.as_ptr(),
                self.input_names.len(),
                self.output_names.as_ptr(),
                self.output_names.len(),
                output_values.as_mut_ptr(),
            )
        };
        API.verify_status("Run", status_ptr);

        output_values.into_iter().map(|x| Tensor::new(x)).collect()
    }
}

pub struct MemoryInfo(*mut OrtMemoryInfo);

impl Drop for MemoryInfo {
    fn drop(&mut self) {
        unsafe { (API.release_memory_info_fn)(self.0) };
        self.0 = 0 as *mut OrtMemoryInfo;
    }
}

impl MemoryInfo {
    pub fn create_state_tensor(self: &Self, state_tensors: &mut [StateTensor<f32>]) -> Tensor {
        let shape = state_tensors.shape();
        let size = mem::size_of_val(state_tensors);
        let data_ptr: &mut [f32] = bytemuck::cast_slice_mut(state_tensors);
        API.create_tensor_with_data_as_ort_value(
            self,
            data_ptr.as_mut_ptr() as *mut c_void,
            size,
            shape.as_ptr(),
            shape.len(),
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        )
    }
}

#[derive(Debug)]
pub struct Tensor {
    ptr: *mut OrtValue,
    element_type: ONNXTensorElementDataType,
    shape: Vec<i64>,
}

impl Drop for Tensor {
    fn drop(&mut self) {
        unsafe { (API.release_value_fn)(self.ptr) };
        self.ptr = 0 as *mut OrtValue;
    }
}

impl Tensor {
    fn new(ptr: *mut OrtValue) -> Self {
        let mut type_shape_ptr = 0 as *mut OrtTensorTypeAndShapeInfo;
        let status_ptr = unsafe { (API.get_tensor_type_and_shape)(ptr, &mut type_shape_ptr) };

        API.verify_status("GetTensorTypeAndShape", status_ptr);

        let type_and_shape = TensorTypeAndShape::new(type_shape_ptr);
        Self {
            ptr: ptr,
            element_type: type_and_shape.0,
            shape: type_and_shape.1,
        }
    }
    fn get_shape(&self) -> &Vec<i64> {
        &self.shape
    }

    fn copy_to<T: 'static>(self: &Self, dst: &mut [T]) {
        let type_id = TypeId::of::<T>();
        let expected_type_id = match self.element_type {
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => TypeId::of::<f32>(),
            _ => {
                unimplemented!(
                    "Unsupported ONNXTensorElementDataType({})",
                    self.element_type
                );
            }
        };

        if type_id != expected_type_id {
            panic!(
                "{:?} is expected but {:?} is supplied",
                expected_type_id, type_id
            );
        }

        assert_eq!(
            dst.len(),
            self.shape.iter().copied().reduce(|a, b| a * b).unwrap() as usize
        );

        let mut src = 0 as *mut c_void;
        let status_ptr = unsafe { (API.get_tensor_mutable_data_fn)(self.ptr, &mut src) };
        API.verify_status("GetTensorMutableData", status_ptr);

        unsafe {
            ptr::copy::<T>(src as *const T, dst.as_mut_ptr(), dst.len());
        }
    }
}

#[derive(Debug)]
pub struct TypeInfo(*mut OrtTypeInfo);

impl Drop for TypeInfo {
    fn drop(&mut self) {
        unsafe { (API.release_type_info_fn)(self.0) };
        self.0 = 0 as *mut OrtTypeInfo;
    }
}

impl TypeInfo {
    fn get_type_and_shape(&self) -> TensorTypeAndShape {
        let mut ptr = 0 as *const OrtTensorTypeAndShapeInfo;
        let status_ptr = unsafe { (API.cast_type_info_to_tensor_info_fn)(self.0, &mut ptr) };
        API.verify_status("CastTypeInfoToTensorInfo", status_ptr);

        TensorTypeAndShape::new(ptr)
    }
}

#[derive(Debug)]
pub struct TensorTypeAndShape(ONNXTensorElementDataType, Vec<i64>);

impl TensorTypeAndShape {
    fn new(ptr: *const OrtTensorTypeAndShapeInfo) -> Self {
        let mut element_type: ONNXTensorElementDataType =
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        let status_ptr = unsafe { (API.get_tensor_element_type_fn)(ptr, &mut element_type) };
        API.verify_status("GetTensorElementType", status_ptr);

        let mut dims_count: usize = 0;
        let status_ptr = unsafe { (API.get_dimensions_count_fn)(ptr, &mut dims_count) };
        API.verify_status("GetDimensionsCount", status_ptr);

        let mut dims = vec![0i64; dims_count];
        let status_ptr = unsafe { (API.get_dimensions_fn)(ptr, dims.as_mut_ptr(), dims_count) };
        API.verify_status("GetDimensions", status_ptr);
        Self(element_type, dims)
    }
}

#[test]
fn test_ort() {
    let mut session_options = API.create_session_options();
    session_options.set_intra_op_num_threads(1);
    session_options.set_session_graph_optimization_level(GraphOptimizationLevel_ORT_ENABLE_ALL);

    let memory_info = API.create_cpu_memory_info(
        OrtAllocatorType_OrtArenaAllocator,
        OrtMemType_OrtMemTypeDefault,
    );

    let mut state_tensor = StateTensor::<f32>::default();
    state_tensor[1][7][7] = 1f32;
    state_tensor[2][7][7] = 1f32;
    let mut state_tensor_batch = vec![state_tensor];
    let input_tensor = memory_info.create_state_tensor(&mut state_tensor_batch);

    let session = API.create_session("model.onnx", &session_options);
    let output_tensors = session.run(&input_tensor);

    assert_eq!(output_tensors.len(), 2);

    let mut prob_matrix: SquareMatrix<f32> = SquareMatrix::default();
    output_tensors[0].copy_to::<f32>(bytemuck::cast_slice_mut(&mut prob_matrix));
    println!("{:?}", &prob_matrix);

    let mut score = [0f32; 1];
    output_tensors[1].copy_to::<f32>(bytemuck::cast_slice_mut(&mut score));
    println!("{}", score[0]);
}
