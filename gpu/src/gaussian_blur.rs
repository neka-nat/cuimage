#![cfg_attr(
    target_os = "cuda",
    no_std,
    feature(register_attr),
    register_attr(nvvm_internal)
)]

use cuda_std::prelude::*;

#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn gaussian_blur(
    src: &[u8],
    weights: &[f32],
    dst: *mut u8,
    width: u32,
    height: u32,
    num_of_channels: u32,
    sigma: i32,
) {
    let idx = thread::index_1d() as usize;
    let output_num_of_channels = 3;
    if idx < src.len() {
        let num_of_channels = num_of_channels as usize;
        let x = idx % width as usize;
        let y = idx / width as usize;
        let mut r_sum = 0.0;
        let mut g_sum = 0.0;
        let mut b_sum = 0.0;
        for i in -sigma..(sigma + 1) {
            for j in -sigma..(sigma + 1) {
                if x as i32 + i < 0
                    || x as i32 + i >= width as i32
                    || y as i32 + j < 0
                    || y as i32 + j >= height as i32
                {
                    continue;
                }
                let k = (i + sigma + (j + sigma) * (sigma * 2 + 1)) as usize;
                let x_i = (x as i32 + i) as usize;
                let y_j = (y as i32 + j) as usize;
                let l = num_of_channels * (y_j * width as usize + x_i);
                r_sum += weights[k] * src[l] as f32;
                g_sum += weights[k] * src[l + 1] as f32;
                b_sum += weights[k] * src[l + 2] as f32;
            }
        }
        let r_elem = &mut *dst.add(output_num_of_channels * idx);
        let g_elem = &mut *dst.add(output_num_of_channels * idx + 1);
        let b_elem = &mut *dst.add(output_num_of_channels * idx + 2);
        *r_elem = r_sum as u8;
        *g_elem = g_sum as u8;
        *b_elem = b_sum as u8;
    }
}
