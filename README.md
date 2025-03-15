# Machine Learning Algorithm for Character Recognition (10x10 Pixels)

This Python-based machine learning algorithm is designed to recognize a limited set of characters formed by 10x10 pixel images. The code performs a defined number of iterations for training and then carries out inference with slight alterations in the characters. It also generates a learning curve graph and a confusion matrix after evaluation.

## Overview

This project implements a character recognition system using a simple machine learning model. The algorithm is trained on 10x10 pixel images representing characters and then evaluated using slightly altered test images to assess the model's generalization ability. After training, the following outputs are generated:
- Learning curve
- Confusion matrix

## Features
- Recognition of characters formed by 10x10 pixels.
- Iterative training to optimize model performance.
- Inference with slight variations in the characters.
- Visualization of learning curve.
- Generation of confusion matrix to assess performance.

## Installation

To use this project, make sure to have Python 3.12.2 installed along with the necessary libraries. You can install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

### Requirements
- Python 3.12.2
- Seaborn
- Pandas
- NumPy
- Matplotlib

## Usage

1. **Training the Model**:  
   Run the `main.py` script to train the model with the predefined dataset of 10x10 pixel characters.

   ```bash
   python main.py
   ```

2. **Inference**:  
   After training, the model will make predictions on test data with slight modifications to the characters.

3. **Output**:  
   - A graph displaying the learning curve will be shown.
   - A confusion matrix will be printed, showing the model's performance across various classes.

## Code Structure

- `main.py`: The main script to train the model and generate visualizations.
- `Images/`: Folder containing the dataset (training and testing images).
- `multiclass.py`: Contains the machine learning model architecture and training functions.

## Example Output

![image](https://github.com/user-attachments/assets/a5738769-d99d-4c3c-a5b9-220da5689268)
