from metaflow import FlowSpec, step


class BatchCollectionFlow(FlowSpec):
    """
    A flow for collecting or updating data in a batch process.
    """

    @step
    def start(self):
        """
        This is the 'start' step. All flows must have a step named 'start' that
        is the first step in the flow.
        """
        print("Batch Collection Flow is starting.")

        self.next(self.load_inputs)

    @step
    def load_inputs(self):
        """
        This step gathers the list of stocks to be collected and makes sure everything needed is found.

        """

        import pandas as pd
        import os
        from pathlib import Path

        base_data_path = os.environ["BASE_DATA_PATH"]

        tickers_path = Path(base_data_path)/"tickers.csv"

        use_cols = [
            'Symbol',
            'Name',
            'Country',
            'IPO Year',
            'Sector',
            'Industry'
        ]

        new_col_names = [_.replace(' ', '_').lower() for _ in use_cols]

        tickers = pd.read_csv(tickers_path, usecols = use_cols)

        tickers.rename(columns = {old_col:new_col for old_col, new_col in zip(use_cols, new_col_names)}, inplace = True)

        # TODO: Remove later, this is temporary for testing
        tickers = tickers[:10]
        
        #Check if metadata file exists:
        metadata_path = Path(base_data_path) / "metadata.csv"
        
        if metadata_path.exists():
            #TODO add code to validate and extract metadata in a list of tuples
            metadata_info = []

        else:
            metadata_info = []

        self.metadata_info = metadata_info

        self.next(self.build_collection_plan)


    @step
    def build_collection_plan(self):

        #TODO implement collection plans
        self.next(self.collect_data)

    @step
    def collect_data(self):
        """
        This is the 'end' step. All flows must have an 'end' step, which is the
        last step in the flow.

        """

        # Example run
        # from stock_market_analytics.data_collection import (
        #     YFinanceCollector,
        #     ContinuousTimelineProcessor,
        # )

        # collector = YFinanceCollector(symbol = 'NVDA', start = '2025-01-01')

        # df = collector.get_historical_data()

        # processor = ContinuousTimelineProcessor('NVDA', df)
        # df2 = processor.process()

        # print(df2.head())

        self.next(self.end)

    @step
    def end(self):
        """
        This is the 'end' step. All flows must have an 'end' step, which is the
        last step in the flow.

        """
        print("BatchCollectionFlow is all done.")


if __name__ == "__main__":
    BatchCollectionFlow()