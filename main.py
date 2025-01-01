from data_extractor import DataExtractor
from data_extractor import DataExtractorType

if __name__ == '__main__':
    data_extractor = DataExtractor(
        "imgs/tik_2.jpg", DataExtractorType.tiktok)
    data_extractor.extract()
