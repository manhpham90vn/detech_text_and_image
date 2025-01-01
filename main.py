from data_extractor import DataExtractor
from data_extractor import DataExtractorType

if __name__ == '__main__':
    data_extractor = DataExtractor(
        "imgs/ig_2.jpg", DataExtractorType.instagram)
    data_extractor.extract()
