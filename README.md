# NLP-Tea

*A practical repository combining NLP theory and code, organized for easy learning and collaboration.*
---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [License](#license)
- [Contributing](#contributing)

## Introduction

DerivativeXpert is a comprehensive tool designed to assist with various calculus operations. Whether you are a student, educator, or professional, DerivativeXpert provides functionalities for basic calculations, advanced calculus operations, and plotting.

## Features

- **Basic Calculations**: Perform basic arithmetic and algebraic operations.
- **Advanced Calculus**: Compute derivatives, slope, concavity, and more.
- **Plotting**: Visualize functions and their derivatives.
- **Documentation**: Well-documented code with docstrings for all major classes and methods.
- **Code Style**: Ensures code adheres to PEP 8 standards using pycodestyle.

## Installation

To install DerivativeXpert, clone the repository and install the required dependencies:

```sh
git clone https://github.com/Ad7amstein/DerivativeXpert.git
cd DerivativeXpert
pip install -r requirements.txt
```

## Usage

To use DerivativeXpert, run the main.py script:

```sh
python main.py
```

You can also import and use the modules directly in your own scripts:

```py
from basic_calculator.basic_calc import BasicCalculator
from advanced_calc.derivative import Derivative
```

### Example usage

```py
calc = BasicCalculator("ln(E^2)+3")
result = calc.evaluate()
print(result)

func = Function("x^2 + 3*x + 2")
derivative_result = func.differentiate()
print(derivative_result)
```

## Testing

To run the tests, use the unittest framework:

```sh
python -m unittest discover tests
```

The tests cover various aspects of the project, including basic calculations, advanced calculus operations, and code style checks.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure that your code adheres to the project's coding standards and includes appropriate tests.

---

Thank you for using DerivativeXpert! If you have any questions or feedback, feel free to open an issue on GitHub.
