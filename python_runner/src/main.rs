extern crate clap;
use clap::{Args, Parser, Subcommand};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::fs::OpenOptions;
use std::io::prelude::*;

static ABOUT_TEXT: &str = "Python runner ";

/// Python Runner
#[derive(Parser, Debug)]
#[clap(author, version, about = ABOUT_TEXT, long_about = Some(ABOUT_TEXT), trailing_var_arg=true)]
struct Arguments {
    /// Python source file
    #[clap(display_order = 1, short, long)]
    filename: String,
}

fn main() -> PyResult<()> {
    let args = Arguments::parse();

    let mut file = OpenOptions::new()
        .read(true)
        .write(false)
        .truncate(false)
        .create(false)
        .open(&args.filename)?;

    let mut source_code = String::new();
    file.read_to_string(&mut source_code)?;

    let arg1 = "arg1";
    let arg2 = "arg2";
    let arg3 = "arg3";

    Python::with_gil(|py| {
        let module = PyModule::from_code(py, &source_code, "", "")?;
        let predict_fn: Py<PyAny> = module.getattr("predict")?.into();

        /*
        // call object without any arguments
        fun.call0(py)?;

        // call object with PyTuple
        let args = PyTuple::new(py, &[arg1, arg2, arg3]);
        fun.call1(py, args)?;

        // pass arguments as rust tuple
        let args = (arg1, arg2, arg3);
        fun.call1(py, args)?;
        */
        Ok(())
    })
}
