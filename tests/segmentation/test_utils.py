from telesto.instance_segmentation import rle_encode


def test_rle_encode_1_3_lines():
    size = (3, 3)
    # XY pairs
    coords = [
        (0, 0), (1, 0), (2, 0),
        (0, 2), (1, 2), (2, 2)
    ]

    rle_str = rle_encode(coords, size)

    assert rle_str == "0 3 6 3"


def test_rle_encode_2_line():
    size = (3, 3)
    # XY pairs
    coords = [
        (0, 1), (1, 1), (2, 1),
    ]

    rle_str = rle_encode(coords, size)

    assert rle_str == "3 3"


def test_rle_encode_empty():
    size = (3, 3)
    # XY pairs
    coords = []

    rle_str = rle_encode(coords, size)

    assert rle_str == ""


def test_rle_encode_one_pixel():
    size = (3, 3)
    # XY pairs
    coords = [(1, 1)]

    rle_str = rle_encode(coords, size)

    assert rle_str == "4 1"


def test_rle_encode_two_pixels():
    size = (3, 3)
    # XY pairs
    coords = [(0, 0), (2, 2)]

    rle_str = rle_encode(coords, size)

    assert rle_str == "0 1 8 1"
