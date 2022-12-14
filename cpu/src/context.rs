use cust::prelude::*;

static PTX: &str = include_str!("../../resources/cuimage_gpu.ptx");

pub struct CuContext {
    pub module: Module,
    pub stream: Stream,
    _ctx: Context,
}

impl Default for CuContext {
    fn default() -> Self {
        let ctx = cust::quick_init().unwrap();
        let module = Module::from_ptx(PTX, &[]).unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        Self {
            module,
            stream,
            _ctx: ctx,
        }
    }
}
