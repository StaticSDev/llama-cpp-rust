use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo must set OUT_DIR"));
    let archive = out_dir.join("liblcrcuda.a");

    let status = Command::new("nvcc")
        .args(["-O3", "-arch=sm_75", "-lib", "-o"])
        .arg(&archive)
        .arg("cuda/lcr_kernels.cu")
        .status();

    match status {
        Ok(status) if status.success() => {}
        Ok(status) => fail(&format!("nvcc exited with {status}")),
        Err(e) => fail(&format!("could not run nvcc: {e}")),
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=lcrcuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rerun-if-changed=cuda/lcr_kernels.cu");
    println!("cargo:rerun-if-changed=cuda/lcr_kernels.cuh");
}

fn fail(reason: &str) -> ! {
    println!("cargo:warning=CUDA unavailable: {reason}; build with default features for CPU only");
    panic!("CUDA is unavailable: {reason}. Install nvcc and CUDA Runtime, or build without --features cuda");
}
