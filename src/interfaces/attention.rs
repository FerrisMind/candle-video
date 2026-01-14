use std::sync::Arc;

use candle_core::{Result, Tensor};

pub trait AttentionModule: Send + Sync {
    fn forward_internal(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        image_rotary_emb: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor>;
}

pub trait AttnProcessor: Send + Sync + std::fmt::Debug {
    fn process(
        &self,
        attention: &dyn AttentionModule,
        hidden_states: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        image_rotary_emb: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor>;
}

#[derive(Debug, Default)]
pub struct DefaultAttnProcessor;

impl AttnProcessor for DefaultAttnProcessor {
    fn process(
        &self,
        attention: &dyn AttentionModule,
        hidden_states: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        image_rotary_emb: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        attention.forward_internal(
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            image_rotary_emb,
        )
    }
}

pub trait AttentionModuleMixin: AttentionModule {
    fn set_processor(&mut self, processor: Arc<dyn AttnProcessor>);
    fn processor(&self) -> &Arc<dyn AttnProcessor>;
}

pub trait AttentionMixin {
    fn set_attn_processor(&mut self, processor: Arc<dyn AttnProcessor>);
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[derive(Debug)]
    struct DummyModule {
        processor: Arc<dyn AttnProcessor>,
    }

    impl DummyModule {
        fn new() -> Self {
            Self {
                processor: Arc::new(DefaultAttnProcessor),
            }
        }
    }

    impl AttentionModule for DummyModule {
        fn forward_internal(
            &self,
            hidden_states: &Tensor,
            _encoder_hidden_states: Option<&Tensor>,
            _attention_mask: Option<&Tensor>,
            _image_rotary_emb: Option<(&Tensor, &Tensor)>,
        ) -> Result<Tensor> {
            Ok(hidden_states.clone())
        }
    }

    impl AttentionModuleMixin for DummyModule {
        fn set_processor(&mut self, processor: Arc<dyn AttnProcessor>) {
            self.processor = processor;
        }

        fn processor(&self) -> &Arc<dyn AttnProcessor> {
            &self.processor
        }
    }

    #[test]
    fn default_processor_forwards_inputs() {
        let device = Device::Cpu;
        let hidden_states = Tensor::zeros((1, 2, 3), DType::F32, &device).unwrap();
        let module = DummyModule::new();
        let output = module
            .processor()
            .process(&module, &hidden_states, None, None, None)
            .unwrap();
        assert_eq!(output.dims(), hidden_states.dims());
    }
}
