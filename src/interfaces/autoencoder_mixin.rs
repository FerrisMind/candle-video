pub trait AutoencoderMixin {
    fn enable_tiling(&mut self);
    fn disable_tiling(&mut self);
    fn enable_slicing(&mut self);
    fn disable_slicing(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct DummyAutoencoder {
        tiling: bool,
        slicing: bool,
    }

    impl AutoencoderMixin for DummyAutoencoder {
        fn enable_tiling(&mut self) {
            self.tiling = true;
        }

        fn disable_tiling(&mut self) {
            self.tiling = false;
        }

        fn enable_slicing(&mut self) {
            self.slicing = true;
        }

        fn disable_slicing(&mut self) {
            self.slicing = false;
        }
    }

    #[test]
    fn toggles_tiling_and_slicing() {
        let mut ae = DummyAutoencoder::default();
        assert!(!ae.tiling);
        assert!(!ae.slicing);

        ae.enable_tiling();
        ae.enable_slicing();
        assert!(ae.tiling);
        assert!(ae.slicing);

        ae.disable_tiling();
        ae.disable_slicing();
        assert!(!ae.tiling);
        assert!(!ae.slicing);
    }
}
