from data_extractor import DataExtractor
from data_extractor import DataExtractorType

if __name__ == '__main__':
    data_extractor = DataExtractor(
        "imgs/xhs_1.jpeg", DataExtractorType.xhs)
    data_extractor.extract()
