import numpy as np


def my_dct(gray_mat: np.uint8) -> np.array:
    """ Двумерное дискретное косинусное преобразование """
    assert gray_mat.shape == (8, 8)
    assert gray_mat.dtype == np.uint8
    # Создаем матрицу коэффициентов
    coeffs = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            if i == 0:
                coeffs[i, j] = 1 / np.sqrt(8)
            else:
                coeffs[i, j] = np.sqrt(2 / 8) * np.cos((np.pi * (2 * j + 1) * i) / (2 * 8))

    # Вычисляем DCT для блока
    dct_block = np.dot(np.dot(coeffs, gray_mat), coeffs.T)
    return dct_block


def my_idct(dct_block: np.array) -> np.array:
    """Inverse two-dimensional discrete cosine transform"""
    assert dct_block.shape == (8, 8)
    coeffs = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            if i == 0:
                coeffs[i, j] = 1 / np.sqrt(8)
            else:
                coeffs[i, j] = np.sqrt(2 / 8) * np.cos((np.pi * i * (2 * j + 1)) / (2 * 8))
    idct_block = np.dot(np.dot(coeffs.T, dct_block), coeffs)
    idct_block = np.round(idct_block).astype(np.uint8)
    return idct_block
