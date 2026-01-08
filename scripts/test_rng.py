import math

# --- Pcg32 Implementation ---
class Pcg32:
    def __init__(self, seed, inc):
        self.state = 0
        self.inc = (inc << 1) | 1
        self.next_u32()
        self.state = (self.state + seed) & 0xFFFFFFFFFFFFFFFF
        self.next_u32()

    def next_u32(self):
        oldstate = self.state
        self.state = (oldstate * 6364136223846793005 + self.inc) & 0xFFFFFFFFFFFFFFFF
        xorshifted = ((oldstate >> 18) ^ oldstate) >> 27
        rot = (oldstate >> 59)
        xorshifted = xorshifted & 0xFFFFFFFF
        return ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & 0xFFFFFFFF

    def next_f32(self):
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

# Test
rng = Pcg32(42, 1442695040888963407)

print("Python Pcg32 first 10 Gaussian values:")
values = []
for i in range(5):
    z0, z1 = rng.next_gaussian()
    values.append(z0)
    values.append(z1)

print(values[:10])
print(f"mean: {sum(values[:10])/10:.6f}")
