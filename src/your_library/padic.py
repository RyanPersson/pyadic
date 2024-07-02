import numpy as np
from typing import List, Union

class PadicInteger:
    def __init__(self, value: Union[int, List[int]], prime: int, precision: int):
        """
        Initialize a p-adic integer.

        :param value: Either an integer or a list of p-adic digits
        :param prime: The prime base p
        :param precision: The number of p-adic digits to use
        """
        self.prime = prime
        self.precision = precision

        # Choose the smallest possible dtype that can represent digits 0 to p-1
        self.dtype = np.uint8 if prime < 256 else (np.uint16 if prime < 65536 else np.uint32)

        if isinstance(value, int):
            self.digits = self._int_to_padic(value)
        elif isinstance(value, list):
            self.digits = np.array(value[:precision], dtype=self.dtype)
        else:
            raise ValueError("Value must be an integer or a list of digits")

    def _int_to_padic(self, n: int) -> np.ndarray:
        """Convert an integer to its p-adic representation."""
        digits = []
        for _ in range(self.precision):
            digits.append(n % self.prime)
            n //= self.prime
        return np.array(digits, dtype=self.dtype)

    def __repr__(self) -> str:
        return f"PadicInteger({list(self.digits)}, prime={self.prime}, precision={self.precision})"

    def __str__(self) -> str:
        return f"(...{'.'.join(str(d) for d in reversed(self.digits))})"

    def __add__(self, other: 'PadicInteger') -> 'PadicInteger':
        if self.prime != other.prime:
            raise ValueError("Cannot add p-adic integers with different primes")
        
        result = []
        carry = 0
        for i in range(max(self.precision, other.precision)):
            a = self.digits[i] if i < self.precision else 0
            b = other.digits[i] if i < other.precision else 0
            digit_sum = a + b + carry
            result.append(digit_sum % self.prime)
            carry = digit_sum // self.prime
        
        return PadicInteger(result, self.prime, max(self.precision, other.precision))

    def __sub__(self, other: 'PadicInteger') -> 'PadicInteger':
        if self.prime != other.prime:
            raise ValueError("Cannot subtract p-adic integers with different primes")
        
        result = []
        borrow = 0
        for i in range(max(self.precision, other.precision)):
            a = self.digits[i] if i < self.precision else 0
            b = other.digits[i] if i < other.precision else 0
            digit_diff = a - b - borrow + self.prime
            result.append(digit_diff % self.prime)
            borrow = 1 if digit_diff < self.prime else 0
        
        return PadicInteger(result, self.prime, max(self.precision, other.precision))

    def __mul__(self, other: 'PadicInteger') -> 'PadicInteger':
        if self.prime != other.prime:
            raise ValueError("Cannot multiply p-adic integers with different primes")
        
        result = [0] * (self.precision + other.precision)
        for i in range(self.precision):
            carry = 0
            for j in range(other.precision):
                temp = result[i+j] + self.digits[i] * other.digits[j] + carry
                result[i+j] = temp % self.prime
                carry = temp // self.prime
            result[i+other.precision] = carry
        
        return PadicInteger(result, self.prime, self.precision + other.precision)

    # Additional methods can be added here (division, comparison, etc.)

# Example usage
if __name__ == "__main__":
    a = PadicInteger(123, 5, 10)
    b = PadicInteger(456, 5, 10)
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}")
    print(f"a - b = {a - b}")
    print(f"a * b = {a * b}")