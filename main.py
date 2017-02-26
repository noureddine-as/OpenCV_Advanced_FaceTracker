from processing_module import *

# Create a list of available cascades in this form
#  FILE                     NAME        DESC
#  path of the XML file     the name    description
if __name__ == '__main__':
    v = Interface("github.com/noureddine-as", CAM_CODE=0, FLIP_CODE=False)
    c = Controller(interface=v)
    v.detectndraw()

    del v
