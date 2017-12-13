# main.py
from tests_ch03 import test_batch_CH03
from tests_ch04 import test_batch_CH04
from tests_ch05 import test_batch_CH05
from tests_ch06 import test_batch_CH06
from tests_ch08 import test_compression
from tests_ch09 import  test_batch_CH09
import fft_manager

def main():

    # test_batch_CH03()
    # test_batch_CH04()
    # test_batch_CH05()
    # test_batch_CH06()
    # test_compression()
    # test_batch_CH09()
    fft_manager.run()

if __name__ == "__main__":
    main()
