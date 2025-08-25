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

        tickers_path = Path(base_data_path) / "tickers.csv"

        assert tickers_path.exists(), (
            f"Tickers file not found at {tickers_path}. A set of stocks to collect must be provided"
        )

        use_cols = ["Symbol", "Name", "Country", "IPO Year", "Sector", "Industry"]

        new_col_names = [_.replace(" ", "_").lower() for _ in use_cols]

        tickers = pd.read_csv(tickers_path, usecols=use_cols)

        tickers.rename(
            columns={
                old_col: new_col for old_col, new_col in zip(use_cols, new_col_names)
            },
            inplace=True,
        )

        # TODO: Remove later, this is temporary for testing
        tickers = tickers[:10]
        self.tickers = tickers.to_dict(orient="records")

        # Check if metadata file exists:
        metadata_path = Path(base_data_path) / "metadata.csv"

        if metadata_path.exists():
            metadata_df = pd.read_csv(metadata_path)
            assert not metadata_df.empty, "Metadata file is empty"
            assert "symbol" in metadata_df.columns, (
                "Metadata file is missing 'symbol' column"
            )
            assert "last_ingestion" in metadata_df.columns, (
                "Metadata file is missing 'last_ingestion' column"
            )
            assert "max_date_recorded" in metadata_df.columns, (
                "Metadata file is missing 'max_date_recorded' column"
            )
            assert "status" in metadata_df.columns, (
                "Metadata file is missing 'status' column"
            )

            # ensure coltypes:
            metadata_df["last_ingestion"] = pd.to_datetime(
                metadata_df["last_ingestion"], errors="coerce"
            )
            metadata_df["max_date_recorded"] = pd.to_datetime(
                metadata_df["max_date_recorded"], errors="coerce"
            )

            metadata_info = metadata_df.to_dict(orient="records")

        else:
            metadata_info = []

        self.metadata_info = metadata_info

        self.next(self.build_collection_plan)

    @step
    def build_collection_plan(self):
        import pandas as pd

        metadata_info = self.metadata_info
        tickers = self.tickers
        collection_plans = []

        if len(metadata_info) == 0:
            # First-time ingestion, collect maximal history
            for ticker in tickers:
                collection_plans.append({
                    "symbol": ticker["symbol"],
                    "period": "max",
                })

        else:
            for ticker in tickers:
                symbol = ticker["symbol"]
                matching_metadata = next(
                    (item for item in metadata_info if item["symbol"] == symbol), None
                )

                if matching_metadata is None:
                    # New stock, collect maximal history
                    collection_plans.append({"symbol": symbol, "period": "max"})
                else:
                    # Existing stock, collect incremental updates
                    last_ingestion = matching_metadata["last_ingestion"]
                    if matching_metadata["status"] == "active":
                        collection_plans.append(
                            {
                                "symbol": symbol,
                                "start": (last_ingestion + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                            }
                        )

        self.collection_plans = collection_plans

        # parallelizing operations
        self.next(self.collect_data, foreach="collection_plans")

    @step
    def collect_data(self):
        """
        This step collects data for a single stock based on the provided collection plan.
        """

        import pandas as pd

        from stock_market_analytics.data_collection import (
            YFinanceCollector,
            ContinuousTimelineProcessor,
        )

        collection_plan = self.input
        symbol = collection_plan['symbol']

        new_metadata = {
            "symbol": symbol,
            "last_ingestion": pd.Timestamp.now()
            .normalize()
            .date()
            .strftime("%Y-%m-%d"),
        }

        collector = YFinanceCollector(**collection_plan)
        df = collector.get_historical_data()

        if collector.collection_successful and not df.is_empty():
            processor = ContinuousTimelineProcessor(symbol, df)
            df = processor.process()

            if df is not None and processor.processing_successful:
                new_metadata["max_date_recorded"] = (
                    df["date"].max().strftime("%Y-%m-%d")
                )
                new_metadata["status"] = "active"

            else:
                new_metadata["max_date_recorded"] = None
                new_metadata["status"] = "data_issue"
                df = None

        else:
            new_metadata["max_date_recorded"] = None
            new_metadata["status"] = "collection_issue"
            df = None

        self.results = {"data": df, "new_metadata": new_metadata}

        self.next(self.join_results)

    @step
    def join_results(self, inputs):
        print("Joining results from all collection steps.")
        """
        This step joins the results from all data collection steps.
        """
        _results = [input.results for input in inputs]

        import pandas as pd
        from pathlib import Path
        import polars as pl
        import os

        previous_metadata_path = Path(os.environ["BASE_DATA_PATH"]) / "metadata.csv"

        all_collected_data = []
        all_new_metadata = []

        for result in _results:
            if result["data"] is not None and not result["data"].is_empty():
                all_collected_data.append(result["data"])

            if result["new_metadata"] is not None:
                all_new_metadata.append(result["new_metadata"])

        # Collect metadata
        new_metadata_df = pd.DataFrame(all_new_metadata)
        
        if previous_metadata_path.exists():
            previous_metadata_df = pd.read_csv(previous_metadata_path)
            metadata_df = pd.concat([previous_metadata_df, new_metadata_df])
        
        else:
            metadata_df = new_metadata_df.copy()

        metadata_df = (
            metadata_df
            .sort_values(by=["symbol", "last_ingestion"], ascending=[True, True])
            .drop_duplicates(subset=["symbol"], keep="last")
            .reset_index(drop=True)
        )
        self.metadata_df = metadata_df

        # Collect data
        base_data_path = os.environ["BASE_DATA_PATH"]
        stocks_history_path = Path(base_data_path) / "stocks_history.parquet"

        new_data = pl.concat(all_collected_data) if all_collected_data else pl.DataFrame()

        if stocks_history_path.exists():
            existing_data = pl.read_parquet(stocks_history_path)
            data = pl.concat([existing_data, new_data])

        else:
            data = new_data

        # cleaning data
        data = (
            data.sort(["symbol", "date"], descending=[False, True])
            .group_by(["symbol", "date"])
            .agg(pl.all().last())
            .sort(["symbol", "date"], descending=[False, True])
        )

        #write the parquet
        data.write_parquet(stocks_history_path)

        metadata_df = self.metadata_df

        metadata_df.to_csv(previous_metadata_path, index = False)

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
