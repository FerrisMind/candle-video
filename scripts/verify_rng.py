import torch
import safetensors.torch
import math
import struct

class Pcg32:
    def __init__(self, seed, inc):
        self.state = 0
        self.inc = (inc << 1) | 1
        self.next_u32()
        self.state = (self.state + seed) & 0xFFFFFFFFFFFFFFFF
        self.next_u32()

    def next_u32(self):
        oldstate = self.state
        # Advance internal state
        self.state = (oldstate * 6364136223846793005 + self.inc) & 0xFFFFFFFFFFFFFFFF
        
        # Calculate output function (XSH-RR)
        xorshifted = ((oldstate >> 18) ^ oldstate) >> 27
        rot = (oldstate >> 59)
        xorshifted = xorshifted & 0xFFFFFFFF
        
        # Rotate right
        return ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & 0xFFFFFFFF

    def next_f32(self):
        # 2^{-24} = 5.9604645e-8
        return (self.next_u32() >> 8) * 5.9604645e-8

    def next_gaussian(self):
        while True:
            x = self.next_f32()
            if x > 1e-7:
                 break
        u1 = x
        u2 = self.next_f32()

        mag = math.sqrt(-2.0 * math.log(u1))
        z0 = mag * math.cos(2.0 * math.pi * u2)
        z1 = mag * math.sin(2.0 * math.pi * u2)
        
        return z0, z1

    def randn(self, shape):
        count = 1
        for d in shape:
            count *= d
            
        data = []
        i = 0
        while i < count:
            z0, z1 = self.next_gaussian()
            data.append(z0)
            if i + 1 < count:
                data.append(z1)
            i += 2
            
        return torch.tensor(data[:count]).reshape(shape)

def main():
    seed = 42
    rng = Pcg32(seed, 1442695040888963407)
    
    print("Generating 10 random samples in Python...")
    tensor = rng.randn((10,))
    
    print("Random values:")
    for v in tensor:
        print(f"{v:.8f}")

    # Verify against Rust output
    try:
        rust_data = safetensors.torch.load_file("rng_rust.safetensors")
        rust_tensor = rust_data["noise"]
        
        diff = (tensor - rust_tensor).abs().max().item()
        print(f"\nComparing with Rust output...")
        print(f"Max Difference: {diff:.10f}")
        
        if diff < 1e-6:
            print("SUCCESS: Python and Rust RNGs match!")
        else:
            print("FAILURE: RNGs do not match!")
            
    except Exception as e:
        print(f"Could not check Rust output (maybe run rust binary first?): {e}")

if __name__ == "__main__":
    main()
