import numpy as np
import os

##def read_amira_header ():
    #"""
    #Standard Header Format: Avizo Binary file
    #-----------------------------------------
    #Line #      Contents
    #------      --------
    #0           # Avizo BINARY-LITTLE-ENDIAN 2.1
    #1           '\n',
    #2           '\n', 
    #3            'define Lattice 426 426 121\n', 
    #4           '\n', 
    #5           'Parameters {\n',    
    #6                        'Units {\n',
    #7                               'Coordinates "m"\n',
    #8                               '}\n',
    #9                        'Colormap "labels.am",\n',
    #10                       'Content "426x426x121 ushort, uniform coordinates",\n',
    #11                       'BoundingBox 1417.5 5880 1407 5869.5 5649 6909,\n',
    #12                       'CoordType "uniform"\n',
    #13                       '}\n',
    #14          '\n', 
    #15          'Lattice { ushort Labels } @1(HxByteRLE,44262998)\n',
    #16          '\n', 
    #17          '# Data section follows\n'
    #"""
##    pass


#Reference am files:
f_path = '/home/giltis/Dropbox/BNL_Docs/Alt_File_Formats/am_cnvrt_compare/'
fname_flt = 'Shew_C5_bio_abv.am' #Grayscale volume: float dtype
fname_short = 'C2_dType_Short.am' #Grayscale volume: short dtype
fname_test = 'APS_2C_Raw_Abv_CROP_tester.am' #Grayscale volume: float dtype
fname_dbasin = 'C2_dBasin.am' #labelfield: ushort dtype
fname_label = 'C2_LabelField.am' #labelfield: ushort dtype
fname_label2 = 'Rad1_blw_GlsBd-Label.surf' #surface file. not sure we can read yet
fname_binary = 'Shew_C8_bio_blw_GlsBd-Bnry.am' #binary data set: byte dtype
#fname_list = [fname_flt, fname_short, fname__test, fname_dbasin, fname_label, fname_label2, fname_binary]
#head_list = [head_flt, head_short, head_test, head_dbasin, head_label, head_label2, head_binary]
#data_list = 
def _read_amira (src_file):
    """
    This function reads all information contained within standard AmiraMesh
    data sets, and separates the header information from actual image, or 
    volume, data. The function then outputs two lists of strings. The first, 
    am_header, contains all of the raw header information. The second, am_data,
    contains the raw image data.
    NOTE: Both function outputs will require additional processing in order to
    be usable in python and/or with the NSLS-2 function library.
    
    Parameters
    ----------
    src_file : string
        The path and file name pointing to the AmiraMesh file to be loaded.
        
    
    Returns
    -------
    am_header : List of strings
        This list contains all of the raw information contained in the AmiraMesh
        file header. Each line of the original header has been read and stored
        directly from the source file, and will need some additional processing
        in order to be useful in the analysis of the data using the NSLS-2 
        image processing function set.
    
    am_data : string
        A compiled string containing all of the image array data, as was stored
        in the AmiraMesh data file.  
    """
    
    am_header = []
    am_data = []
    f = open(os.path.normpath(src_file), 'r')
    while True:
        line = f.readline()
        am_header.append(line)
        if (line == '# Data section follows\n'):
            f.readline()
            break
    am_data = f.read()
    f.close()
    return am_header, am_data

def _cnvrt_amira_data_2numpy (am_data, header_dict, flip_Z = True):
    """
    The standard format for Avizo Binary files
        is IEEE binary. Big or little endian-ness is stipulated in the header
        information, and will be assessed
    
    Parameters
    ----------
    am_data : string
    
    header_dict : md_dict
    
    flip_Z : bool
    
    Returns
    -------
    output : ndarray
    
    """
    Zdim = header_dict['array_dims']['z_dim']
    Ydim = header_dict['array_dims']['y_dim']
    Xdim = header_dict['array_dims']['x_dim']
    #Strip out null characters from the string of binary values
    data_strip = am_data.strip('\n')
    #Dictionary of the encoding types for AmiraMesh files
    am_format_dict = {'BINARY-LITTLE-ENDIAN' : '<',
                      'BINARY' : '>',
                      'ASCII' : 'unknown'
                     }
    #Dictionary of the data types encountered so far in AmiraMesh files
    am_dtype_dict = {'float' : 'f4',
                     'short' : 'h4',
                     'ushort' : 'H4',
                     'byte' : 'b'
                         }
    if header_dict['data_format'] == 'BINARY-LITTLE-ENDIAN':
        flt_values = np.fromstring(data_strip, (am_format_dict[header_dict['data_format']] + am_dtype_dict[header_dict['data_type']])
    #Resize the 1D array to the correct ndarray dimensions
    flt_values.resize(Zdim, Ydim, Xdim)
    if flip_Z == True:
        output = flt_values[::-1, ..., ...]
    else:
        output = flt_values
    return output

def _sort_amira_header (header_list):
    """
    
    Parameters
    ----------
    header_list : list of strings
    
    Returns
    -------
    
    
    
    """
    
    for row in range(len(header_list)):
        header_list[row] = header_list[row].strip('\n')
        header_list[row] = header_list[row].split(" ")
        for column in range(len(header_list[row])):
            header_list[row] = filter(None, header_list[row])
    header_list = filter(None, header_list)
    return header_list

def _create_md_dict (header_list):
    """
    
    
    """
    
        md_dict = {'software_src' : header_list[0][1],
               'data_format' : header_list[0][2],
               'data_format_version' : header_list[0][3]
                }
    for row in range(len(header_list)):
        try:
            md_dict['array_dims'] = {'x_dim' : int(header_list[row][header_list[row].index('define') + 2]),
                                     'y_dim' : int(header_list[row][header_list[row].index('define') + 3]),
                                     'z_dim' : int(header_list[row][header_list[row].index('define') + 4])
                                     }
        except:
        #    continue
            try:
                md_dict['data_type'] = header_list[row][header_list[row].index('Content') + 2]
            except: 
            #    continue
                try:
                    md_dict['coord_type'] = header_list[row][header_list[row].index('CoordType') + 1]
                except:
                    try:
                        md_dict['bounding_box'] = {'x_min' : float(header_list[row][header_list[row].index('BoundingBox') + 1]),
                                                   'x_max' : float(header_list[row][header_list[row].index('BoundingBox') + 2]),
                                                   'y_min' : float(header_list[row][header_list[row].index('BoundingBox') + 3]),
                                                   'y_max' : float(header_list[row][header_list[row].index('BoundingBox') + 4]),
                                                   'z_min' : float(header_list[row][header_list[row].index('BoundingBox') + 5]),
                                                   'z_max' : float(header_list[row][header_list[row].index('BoundingBox') + 6])
                                                   }
                    except:
                        try:
                            md_dict['units'] = header_list[row][header_list[row].index('Units') + 2]
                            md_dict['coordinates'] = header_list[row + 1][1]
                        except:
                            continue
    return md_dict

def load_am_as_np(file_path):
    """
    This function will load and convert an AmiraMesh binary file to a numpy 
    array. All pertinent information contained in the .am header file is written
    to a metadata dictionary, which is returned along with the numpy array 
    containing the image data.
    
    Parameters
    ----------
    file_path : string
        The path and file name of the AmiraMesh file to be loaded.
        
    Returns
    -------
    md_dict : dictionary
        Dictionary containing all pertinent header information associated with 
        the data set.
    
    np_array : float ndarray
        An ndarray containing the image data set to be loaded. Values contained 
        in the resulting volume are set to be of float data type by default.
    """
    
    header, data = __read_amira__(file_path)
    header = __sort_amira_header__(header)
    md_dict = __create_md_dict__(header)
    np_array = __cnvrt_amira_data_2numpy__(data, md_dict)
    return md_dict, np_array
