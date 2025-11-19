import json
from pathlib import Path
from typing import List, Tuple, Dict, Any


class HQSmallDataLoader:
    """
    HQ-small数据集加载器
    """

    def __init__(self, data_dir: str = "./data"):
        """
        初始化数据加载器

        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def load_collection(self, collection_path: str = "./data/collection.jsonl") -> Tuple[List[str], List[str]]:
        """
        加载文档集合

        Args:
            collection_path: 集合文件路径

        Returns:
            文档ID列表和文档内容列表
        """
        doc_ids = []
        documents = []

        with open(collection_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                doc_ids.append(item['id'])
                documents.append(item['text'])

        print(f"Loaded {len(documents)} documents from collection")
        return doc_ids, documents

    def load_train_set(self, train_path: str = "./data/train.jsonl") -> List[Dict[str, Any]]:
        """
        加载训练集

        Args:
            train_path: 训练集文件路径

        Returns:
            训练数据列表，每个元素包含id, text, answer, supporting_ids
        """
        train_data = []

        with open(train_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                train_data.append({
                    'id': item['id'],
                    'text': item['text'],
                    'answer': item['answer'],
                    'supporting_ids': item['supporting_ids']
                })

        print(f"Loaded {len(train_data)} samples from training set")
        return train_data

    def load_validation_set(self, validation_path: str = "./data/validation.jsonl") -> List[Dict[str, Any]]:
        """
        加载验证集

        Args:
            validation_path: 验证集文件路径

        Returns:
            验证数据列表，每个元素包含id, text, answer, supporting_ids
        """
        validation_data = []

        with open(validation_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                validation_data.append({
                    'id': item['id'],
                    'text': item['text'],
                    'answer': item['answer'],
                    'supporting_ids': item['supporting_ids']
                })

        print(f"Loaded {len(validation_data)} samples from validation set")
        return validation_data

    def load_test_set(self, test_path: str = "./data/test.jsonl") -> List[Dict[str, Any]]:
        """
        加载测试集

        Args:
            test_path: 测试集文件路径

        Returns:
            测试数据列表，每个元素包含id, text
        """
        test_data = []

        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                test_data.append({
                    'id': item['id'],
                    'text': item['text']
                })

        print(f"Loaded {len(test_data)} samples from test set")
        return test_data