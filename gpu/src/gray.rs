#![cfg_attr(
    target_os = "cuda",
    no_std,
    feature(register_attr),
    register_attr(nvvm_internal)
)]

use cuda_std::prelude::*;

#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn gray(src: &[u8], dst: *mut u8, num_of_channels: u32) {
    let idx = thread::index_1d() as usize;
    if idx < src.len() {
        let num_of_channels = num_of_channels as usize;
        let r = src[num_of_channels * idx] as f32;
        let g = src[num_of_channels * idx + 1] as f32;
        let b = src[num_of_channels * idx + 2] as f32;
        let gray = 0.299 * r + 0.587 * g + 0.114 * b;
        let elem = &mut *dst.add(idx);
        *elem = gray as u8;
    }
}
