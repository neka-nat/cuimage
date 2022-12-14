use cust::memory::DeviceCopy;
use cust::prelude::*;
use image::{ImageBuffer, Pixel};
use std::error::Error;
use std::marker::PhantomData;

pub struct CuImage<P: Pixel>
where
    <P as Pixel>::Subpixel: DeviceCopy,
{
    pub width: u32,
    pub height: u32,
    pub num_of_channels: u32,
    pub data: DeviceBuffer<<P as Pixel>::Subpixel>,
    _phantom: PhantomData<P>,
}

impl<P: Pixel> CuImage<P>
where
    P: 'static,
    <P as Pixel>::Subpixel: DeviceCopy,
{
    pub fn new(width: u32, height: u32) -> Self {
        let img = ImageBuffer::<P, Vec<<P as Pixel>::Subpixel>>::new(width, height);
        CuImage {
            width: width,
            height: height,
            num_of_channels: P::CHANNEL_COUNT as u32,
            data: img.as_flat_samples().as_slice().as_dbuf().unwrap(),
            _phantom: PhantomData,
        }
    }
    pub fn from_host_image(
        img: &ImageBuffer<P, Vec<<P as Pixel>::Subpixel>>,
    ) -> Result<Self, Box<dyn Error>> {
        let width = img.width();
        let height = img.height();
        let data = img.as_flat_samples().as_slice().as_dbuf()?;
        Ok(Self {
            width,
            height,
            num_of_channels: P::CHANNEL_COUNT as u32,
            data,
            _phantom: PhantomData,
        })
    }
    pub fn to_host_image(
        &self,
    ) -> Result<ImageBuffer<P, Vec<<P as Pixel>::Subpixel>>, Box<dyn Error>> {
        let mut out = ImageBuffer::<P, Vec<P::Subpixel>>::new(self.width, self.height);
        self.data
            .copy_to(&mut out.as_flat_samples_mut().as_mut_slice().as_mut())?;
        Ok(out)
    }
}
