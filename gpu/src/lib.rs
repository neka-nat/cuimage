#![cfg_attr(
    target_os = "cuda",
    no_std,
    feature(register_attr),
    register_attr(nvvm_internal)
)]

mod gaussian_blur;
mod gray;

pub use gaussian_blur::*;
pub use gray::*;
