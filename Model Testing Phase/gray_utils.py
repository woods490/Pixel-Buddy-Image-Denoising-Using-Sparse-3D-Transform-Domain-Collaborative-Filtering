'''
This code implements image denoising algorithm proposed in this paper:

Dabov, K., Foi, A., Katkovnik, V., & Egiazarian, K. (2007). 
Image denoising by sparse 3-D transform-domain collaborative filtering. 
IEEE Transactions on image processing, 16(8), 2080-2095.

*Created on 2016.9.13
*Author: Huang Liu
'''

import cv2
import numpy
from tqdm import tqdm

cv2.setUseOptimized(True)

def init_parameters(sigma, quality):


    # Thresholds used to calculate similarity between blocks
    if sigma <= 40:
        First_Match_threshold, Second_Match_threshold = 2500, 40
    elif sigma > 40:
        First_Match_threshold, Second_Match_threshold = 5000, 350
    
    # Parameters related to the number of matching blocks
    Step1_max_matched_cnt, Step2_max_matched_cnt = 16, 32
    
    # Parameters related to the search window
    Step1_Search_Window, Step2_Search_Window = 39, 39
    
    # Beta parameter
    Beta_Kaiser = 2.0 
    
    # Block sizes based on sigma value
    if 2 <= sigma <= 17:
        Step1_Blk_Size, Step2_Blk_Size = 4, 4
    elif 18 <= sigma <= 32:
        Step1_Blk_Size, Step2_Blk_Size = 6, 6
    elif sigma > 32:
        Step1_Blk_Size, Step2_Blk_Size = 8, 6
    elif sigma < 2:
        Step1_Blk_Size, Step2_Blk_Size = 4, 4
        print(f"Your image is already noiseless (sigma={sigma}). Doing BM3D will reduce the image quality. Stopping the denoising is advised")
    
    # Parameters related to block step and search step
    if quality == 'best':
        Step1_Blk_Step, Step1_Search_Step = 2, 2
        Step2_Blk_Step, Step2_Search_Step = 2, 2
    elif quality == 'fast':
        Step1_Blk_Step, Step1_Search_Step = 3, 3
        Step2_Blk_Step, Step2_Search_Step = 2, 2
    else:
        raise ValueError("Invalid value for 'quality'. Use 'best' or 'fast'.")
    
    return (
        First_Match_threshold, Second_Match_threshold,
        Step1_max_matched_cnt, Step2_max_matched_cnt,
        Step1_Search_Window, Step2_Search_Window,
        Beta_Kaiser, Step1_Blk_Size, Step2_Blk_Size,
        Step1_Blk_Step, Step1_Search_Step, Step2_Blk_Step, 
        Step2_Search_Step
    ) 

def init(img, _blk_size, _Beta_Kaiser):
    """
    This function is used for initialization, 
    returns an array used to record the filtered image and weight, and constructs the Kaiser window
    """
    m_shape = img.shape
    m_img = numpy.matrix(numpy.zeros(m_shape, dtype=float))
    m_wight = numpy.matrix(numpy.zeros(m_shape, dtype=float))
    K = numpy.matrix(numpy.kaiser(_blk_size, _Beta_Kaiser))
    m_Kaiser = numpy.array(K.T * K)            # Construct a Kaiser window
    return m_img, m_wight, m_Kaiser


def Locate_blk(i, j, blk_step, block_Size, width, height):
    '''This function is used to ensure that the current blk does not exceed the image range'''
    if i*blk_step+block_Size < width:
        point_x = i*blk_step
    else:
        point_x = width - block_Size

    if j*blk_step+block_Size < height:
        point_y = j*blk_step
    else:
        point_y = height - block_Size

    m_blockPoint = numpy.array((point_x, point_y), dtype=int)  # The vertices of the current reference image

    return m_blockPoint


def Define_SearchWindow(_noisyImg, _BlockPoint, _WindowSize, Blk_Size):
    """This function returns a tuple (x, y) to define the _Search_Window vertex coordinates"""
    point_x = _BlockPoint[0]  # current x coordinates
    point_y = _BlockPoint[1]  # current y coordinates

    # Get the coordinates of the four vertices of SearchWindow
    LX = point_x+Blk_Size/2-_WindowSize/2     # upper left x
    LY = point_y+Blk_Size/2-_WindowSize/2     # upper left y 
    RX = LX+_WindowSize                       # lower left x
    RY = LY+_WindowSize                       # lower left y

    # Determine whether it has crossed the line
    if LX < 0:   LX = 0
    elif RX > _noisyImg.shape[0]:   LX = _noisyImg.shape[0]-_WindowSize
    if LY < 0:   LY = 0
    elif RY > _noisyImg.shape[0]:   LY = _noisyImg.shape[0]-_WindowSize

    return numpy.array((LX, LY), dtype=int)


def Step1_fast_match(_noisyImg, _BlockPoint, Step1_Blk_Size, Step1_Search_Step, First_Match_threshold, Step1_max_matched_cnt, Step1_Search_Window):
    '''
    * Returns the blocks in the neighborhood that are most similar to the current_block. 
    The returned array contains itself.
    *_noisyImg: Noisy image
    *_BlockPoint: The coordinates and size of the current block
    '''
    (present_x, present_y) = _BlockPoint  # current coordinates
    Blk_Size = Step1_Blk_Size
    Search_Step = Step1_Search_Step
    Threshold = First_Match_threshold
    max_matched = Step1_max_matched_cnt
    Window_size = Step1_Search_Window

    blk_positions = numpy.zeros((max_matched, 2), dtype=int)  # Used to record the location of similar blk
    Final_similar_blocks = numpy.zeros((max_matched, Blk_Size, Blk_Size), dtype=float)

    img = _noisyImg[present_x: present_x+Blk_Size, present_y: present_y+Blk_Size]
    dct_img = cv2.dct(img.astype(numpy.float64))  # Perform block and two-dimensional cosine transformation on the target

    Final_similar_blocks[0, :, :] = dct_img
    blk_positions[0, :] = _BlockPoint

    Window_location = Define_SearchWindow(_noisyImg, _BlockPoint, Window_size, Blk_Size)
    blk_num = (Window_size-Blk_Size)/Search_Step  # Determine the maximum number of similar blks that can be found

    blk_num = int(blk_num)
    (present_x, present_y) = Window_location

    similar_blocks = numpy.zeros((blk_num**2, Blk_Size, Blk_Size), dtype=float)
    m_Blkpositions = numpy.zeros((blk_num**2, 2), dtype=int)
    Distances = numpy.zeros(blk_num**2, dtype=float)  # Record the similarity between each blk and it

    # Start searching in _Search_Window. 
    # The initial version first adopts the traversal search strategy, and returns the most similar blocks here.
    matched_cnt = 0
    for i in range(blk_num):
        for j in range(blk_num):
            tem_img = _noisyImg[present_x: present_x+Blk_Size, present_y: present_y+Blk_Size]
            dct_Tem_img = cv2.dct(tem_img.astype(numpy.float64))
            m_Distance = numpy.linalg.norm((dct_img-dct_Tem_img))**2 / (Blk_Size**2)

            # The data recorded below automatically does not consider itself (because it has already been recorded)
            if m_Distance < Threshold and m_Distance > 0:  # It means that a piece that meets the requirements has been found.
                similar_blocks[matched_cnt, :, :] = dct_Tem_img
                m_Blkpositions[matched_cnt, :] = (present_x, present_y)
                Distances[matched_cnt] = m_Distance
                matched_cnt += 1
            present_y += Search_Step
        present_x += Search_Step
        present_y = Window_location[1]
    Distances = Distances[:matched_cnt]
    Sort = Distances.argsort()

    # Count how many similar blks were found
    if matched_cnt < max_matched:
        Count = matched_cnt + 1
    else:
        Count = max_matched

    if Count > 0:
        for i in range(1, Count):
            Final_similar_blocks[i, :, :] = similar_blocks[Sort[i-1], :, :]
            blk_positions[i, :] = m_Blkpositions[Sort[i-1], :]
    return Final_similar_blocks, blk_positions, Count


def Step1_3DFiltering(_similar_blocks, sigma):
    '''
    *3D transformation and filtering processing
    *_similar_blocks: A group of similar blocks, here is already a representation in the frequency domain
    *The third dimension of _similar_blocks must be taken out in sequence, 
     and then filtered with threshold in the frequency domain, and then inversely transformed.
    '''
    statis_nonzero = 0  # Number of non-zero elements
    m_Shape = _similar_blocks.shape
    Threshold_Hard3D = 2.7*sigma
    # The following piece of code is very time consuming
    for i in range(m_Shape[1]):
        for j in range(m_Shape[2]):
            tem_Vct_Trans = cv2.dct(_similar_blocks[:, i, j])
            tem_Vct_Trans[numpy.abs(tem_Vct_Trans[:]) < Threshold_Hard3D] = 0.
            statis_nonzero += tem_Vct_Trans.nonzero()[0].size
            _similar_blocks[:, i, j] = cv2.idct(tem_Vct_Trans)[0]
    return _similar_blocks, statis_nonzero


def Aggregation_hardthreshold(_similar_blocks, blk_positions, m_basic_img, m_wight_img, _nonzero_num, Count, Kaiser):
    '''
    * Perform weighted accumulation on the stack output after 3D transformation 
    and filtering to obtain the preliminary filtered picture
    *_similar_blocks: A group of similar blocks, here is the representation in the frequency domain
    *For the final array, multiply it by the Caesar window and then output it
    '''
    _shape = _similar_blocks.shape
    if _nonzero_num < 1:
        _nonzero_num = 1
    block_wight = (1./_nonzero_num) * Kaiser
    for i in range(Count):
        point = blk_positions[i, :]
        tem_img = (1./_nonzero_num)*cv2.idct(_similar_blocks[i, :, :]) * Kaiser
        m_basic_img[point[0]:point[0]+_shape[1], point[1]:point[1]+_shape[2]] += tem_img
        m_wight_img[point[0]:point[0]+_shape[1], point[1]:point[1]+_shape[2]] += block_wight


def BM3D_1st_step(_noisyImg, sigma, quality):
    """The first step is basic denoising"""
    # Initialize some parameters:
    (width, height) = _noisyImg.shape   # Get the length and width of the image
    (
        First_Match_threshold, Second_Match_threshold,
        Step1_max_matched_cnt, Step2_max_matched_cnt,
        Step1_Search_Window, Step2_Search_Window,
        Beta_Kaiser, Step1_Blk_Size, Step2_Blk_Size,
        Step1_Blk_Step, Step1_Search_Step
    ) = init_parameters(sigma=sigma, quality=quality)
    block_Size = Step1_Blk_Size         # block size
    blk_step = Step1_Blk_Step           # N block step size sliding
    Width_num = (width - block_Size)/blk_step
    Height_num = (height - block_Size)/blk_step

    # Initialize several arrays
    Basic_img, m_Wight, m_Kaiser = init(_noisyImg, Step1_Blk_Size, Beta_Kaiser)

    # Start processing block by block, +2 is to avoid insufficient edges.
    for i in tqdm(range(int(Width_num+2)), desc="BM3D 1st Step Progress", unit="step"):
        for j in range(int(Height_num+2)):
            # m_blockPoint The vertex of the current reference image
            # Locate_blk function is used to ensure that the current blk does not exceed the image range
            m_blockPoint = Locate_blk(i, j, blk_step, block_Size, width, height)  
            Similar_Blks, Positions, Count = Step1_fast_match(_noisyImg, m_blockPoint)
            Similar_Blks, statis_nonzero = Step1_3DFiltering(Similar_Blks)
            Aggregation_hardthreshold(Similar_Blks, Positions, Basic_img, m_Wight, statis_nonzero, Count, m_Kaiser)
    Basic_img[:, :] /= m_Wight[:, :]
    basic = numpy.matrix(Basic_img, dtype=int)
    basic.astype(numpy.uint8)

    return basic


def Step2_fast_match(_Basic_img, _noisyImg, _BlockPoint, Step2_Blk_Size, Second_Match_threshold, Step2_Search_Step, Step2_max_matched_cnt, Step2_Search_Window):
    '''
    *Fast matching algorithm, returns the blocks in the neighborhood that are most similar 
    to the current_block, and returns basicImg and IMG at the same time
     *_Basic_img: Image after basic denoising
     *_noisyImg: Noisy image
     *_BlockPoint: The coordinates and size of the current block
    '''
    (present_x, present_y) = _BlockPoint  # current coordinates
    Blk_Size = Step2_Blk_Size
    Threshold = Second_Match_threshold
    Search_Step = Step2_Search_Step
    max_matched = Step2_max_matched_cnt
    Window_size = Step2_Search_Window

    blk_positions = numpy.zeros((max_matched, 2), dtype=int)  # Used to record the location of similar blk
    Final_similar_blocks = numpy.zeros((max_matched, Blk_Size, Blk_Size), dtype=float)
    Final_noisy_blocks = numpy.zeros((max_matched, Blk_Size, Blk_Size), dtype=float)

    img = _Basic_img[present_x: present_x+Blk_Size, present_y: present_y+Blk_Size]
    dct_img = cv2.dct(img.astype(numpy.float32))  # Perform block and two-dimensional cosine transformation on the target
    Final_similar_blocks[0, :, :] = dct_img

    n_img = _noisyImg[present_x: present_x+Blk_Size, present_y: present_y+Blk_Size]
    dct_n_img = cv2.dct(n_img.astype(numpy.float32))  # Perform block and two-dimensional cosine transformation on the target
    Final_noisy_blocks[0, :, :] = dct_n_img

    blk_positions[0, :] = _BlockPoint

    Window_location = Define_SearchWindow(_noisyImg, _BlockPoint, Window_size, Blk_Size)
    blk_num = (Window_size-Blk_Size)/Search_Step  # Determine the maximum number of similar blks that can be found
    blk_num = int(blk_num)
    (present_x, present_y) = Window_location

    similar_blocks = numpy.zeros((blk_num**2, Blk_Size, Blk_Size), dtype=float)
    m_Blkpositions = numpy.zeros((blk_num**2, 2), dtype=int)
    Distances = numpy.zeros(blk_num**2, dtype=float)  # Record the similarity between each blk and it

    # Start searching in _Search_Window. The initial version first uses a traversal search strategy, 
    # and returns the most similar blocks here.
    matched_cnt = 0
    for i in range(blk_num):
        for j in range(blk_num):
            tem_img = _Basic_img[present_x: present_x+Blk_Size, present_y: present_y+Blk_Size]
            dct_Tem_img = cv2.dct(tem_img.astype(numpy.float32))
            m_Distance = numpy.linalg.norm((dct_img-dct_Tem_img))**2 / (Blk_Size**2)

            # The data recorded below automatically does not consider itself (because it has already been recorded)
            if m_Distance < Threshold and m_Distance > 0:
                similar_blocks[matched_cnt, :, :] = dct_Tem_img
                m_Blkpositions[matched_cnt, :] = (present_x, present_y)
                Distances[matched_cnt] = m_Distance
                matched_cnt += 1
            present_y += Search_Step
        present_x += Search_Step
        present_y = Window_location[1]
    Distances = Distances[:matched_cnt]
    Sort = Distances.argsort()

    # Count how many similar blks were found
    if matched_cnt < max_matched:
        Count = matched_cnt + 1
    else:
        Count = max_matched

    if Count > 0:
        for i in range(1, Count):
            Final_similar_blocks[i, :, :] = similar_blocks[Sort[i-1], :, :]
            blk_positions[i, :] = m_Blkpositions[Sort[i-1], :]

            (present_x, present_y) = m_Blkpositions[Sort[i-1], :]
            n_img = _noisyImg[present_x: present_x+Blk_Size, present_y: present_y+Blk_Size]
            Final_noisy_blocks[i, :, :] = cv2.dct(n_img.astype(numpy.float64))

    return Final_similar_blocks, Final_noisy_blocks, blk_positions, Count


def Step2_3DFiltering(_Similar_Bscs, _Similar_Imgs, sigma):
    '''
    *Collaborative filtering of 3D Wiener transform
    *_similar_blocks: A group of similar blocks, here is the representation in the frequency domain
    *To take out the third dimension of _similar_blocks in sequence, and then do dct, perform Wiener filtering in the frequency domain, and then do the inverse transformation
    *The returned Wiener_wight is used for subsequent Aggregation
    '''
    m_Shape = _Similar_Bscs.shape
    Wiener_wight = numpy.zeros((m_Shape[1], m_Shape[2]), dtype=float)

    for i in range(m_Shape[1]):
        for j in range(m_Shape[2]):
            tem_vector = _Similar_Bscs[:, i, j]
            tem_Vct_Trans = numpy.matrix(cv2.dct(tem_vector))
            Norm_2 = numpy.float64(tem_Vct_Trans.T * tem_Vct_Trans)
            m_weight = Norm_2/(Norm_2 + sigma**2)
            if m_weight != 0:
                Wiener_wight[i, j] = 1./(m_weight**2 * sigma**2)
            else:
                Wiener_wight[i, j] = 10000
            tem_vector = _Similar_Imgs[:, i, j]
            tem_Vct_Trans = m_weight * cv2.dct(tem_vector)
            _Similar_Bscs[:, i, j] = cv2.idct(tem_Vct_Trans)[0]

    return _Similar_Bscs, Wiener_wight


def Aggregation_Wiener(_Similar_Blks, _Wiener_wight, blk_positions, m_basic_img, m_wight_img, Count, Kaiser):
    '''
    * Perform weighted accumulation on the stack output after 3D transformation and filtering to obtain the preliminary filtered picture
    *_similar_blocks: A group of similar blocks, here is the representation in the frequency domain
    *For the final array, multiply it by the Caesar window and then output it
    '''
    _shape = _Similar_Blks.shape
    block_wight = _Wiener_wight # * Kaiser

    for i in range(Count):
        point = blk_positions[i, :]
        tem_img = _Wiener_wight * cv2.idct(_Similar_Blks[i, :, :]) # * Kaiser
        m_basic_img[point[0]:point[0]+_shape[1], point[1]:point[1]+_shape[2]] += tem_img
        m_wight_img[point[0]:point[0]+_shape[1], point[1]:point[1]+_shape[2]] += block_wight


def BM3D_2nd_step(_basicImg, _noisyImg, sigma, quality):
    '''Step 2. Final estimation: using basic estimation, improved grouping and collaborative Wiener filtering'''
    # Initialize some parameters:
    (width, height) = _noisyImg.shape
    (
        First_Match_threshold, Second_Match_threshold,
        Step1_max_matched_cnt, Step2_max_matched_cnt,
        Step1_Search_Window, Step2_Search_Window,
        Beta_Kaiser, Step1_Blk_Size, Step2_Blk_Size,
        Step1_Blk_Step, Step1_Search_Step, Step2_Blk_Step, 
        Step2_Search_Step
    ) = init_parameters(sigma=sigma, quality=quality)
    block_Size = Step2_Blk_Size
    blk_step = Step2_Blk_Step
    Width_num = (width - block_Size)/blk_step
    Height_num = (height - block_Size)/blk_step

    # Initialize several arrays
    m_img, m_Wight, m_Kaiser = init(_noisyImg, block_Size, Beta_Kaiser)

    for i in tqdm(range(int(Width_num+2)), desc="BM3D 2nd Step Progress", unit="step"):
        for j in range(int(Height_num+2)):
            m_blockPoint = Locate_blk(i, j, blk_step, block_Size, width, height)
            Similar_Blks, Similar_Imgs, Positions, Count = Step2_fast_match(_basicImg, _noisyImg, m_blockPoint, Step2_Blk_Size, Second_Match_threshold, Step2_Search_Step, Step2_max_matched_cnt, Step2_Search_Window)
            Similar_Blks, Wiener_wight = Step2_3DFiltering(Similar_Blks, Similar_Imgs, sigma)
            Aggregation_Wiener(Similar_Blks, Wiener_wight, Positions, m_img, m_Wight, Count, m_Kaiser)
    m_img[:, :] /= m_Wight[:, :]
    Final = numpy.matrix(m_img, dtype=int)
    Final.astype(numpy.uint8)

    return Final
