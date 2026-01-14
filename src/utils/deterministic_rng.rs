use candle_core::{Device, Result, Tensor};

pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    pub fn new(seed: u64, inc: u64) -> Self {
        let mut rng = Self {
            state: 0,
            inc: (inc << 1) | 1,
        };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    pub fn next_u32(&mut self) -> u32 {
        let oldstate = self.state;

        self.state = oldstate
            .wrapping_mul(6364136223846793005)
            .wrapping_add(self.inc);

        let xorshifted = ((oldstate >> 18) ^ oldstate) >> 27;
        let rot = (oldstate >> 59) as u32;
        let xorshifted = xorshifted as u32;
        (xorshifted >> rot) | (xorshifted << ((0u32).wrapping_sub(rot) & 31))
    }

    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * 5.9604645e-8
    }

    pub fn next_gaussian(&mut self) -> (f32, f32) {
        let u1 = loop {
            let x = self.next_f32();
            if x > 1e-7 {
                break x;
            }
        };
        let u2 = self.next_f32();

        let mag = (-2.0 * u1.ln()).sqrt();
        let z0 = mag * (2.0 * std::f32::consts::PI * u2).cos();
        let z1 = mag * (2.0 * std::f32::consts::PI * u2).sin();

        (z0, z1)
    }

    pub fn randn(
        &mut self,
        shape: impl Into<candle_core::Shape>,
        device: &Device,
    ) -> Result<Tensor> {
        let shape = shape.into();
        let elem_count = shape.elem_count();
        let mut data = Vec::with_capacity(elem_count);

        let mut i = 0;
        while i < elem_count {
            let (z0, z1) = self.next_gaussian();
            data.push(z0);
            if i + 1 < elem_count {
                data.push(z1);
            }
            i += 2;
        }

        Tensor::from_vec(data, shape, device)
    }
}
