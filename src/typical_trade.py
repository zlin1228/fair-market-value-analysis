import settings
from generator import TraceDataGenerator
from model import TraceModel
from helpers import epoch_ns_to_iso_str
import pyarrow as pa


# Create buy, sell, dealer row at a given time.
def create_fmv_z_rows(figi_row: pa.Table, ordinals, quantity, execution_dates=None, report_dates=None):
    assert figi_row.shape[0] == 1, "Input table must have exactly one row"
    length = 1 if execution_dates is None else len(execution_dates)

    if not all(key in ordinals for key in ['buy_sell', 'side', 'ats_indicator']):
        raise KeyError("`ordinals` dictionary must contain keys 'buy_sell', 'side', and 'ats_indicator'")

    def make_side_rows(buy_sell, side, ats_indicator):
        d = {
            'buy_sell': [ordinals['buy_sell'].index(buy_sell).as_py()]*length,
            'side': [ordinals['side'].index(side).as_py()]*length,
            'ats_indicator': [ordinals['ats_indicator'].index(ats_indicator).as_py()]*length,
            'quantity': [quantity]*length,
        }
        if report_dates is not None:
            d['report_date'] = report_dates
        if execution_dates is not None:
            d['execution_date'] = execution_dates
        for c in figi_row.column_names:
            if c not in d:
                d[c] = [figi_row[c][0].as_py()]*length
        return pa.Table.from_pydict(d, figi_row.schema).select(figi_row.column_names)

    buy_rows = make_side_rows('B', 'C', 'N')
    sell_rows = make_side_rows('S', 'C', 'N')
    dealer_rows = make_side_rows('S', 'D', 'N')

    return pa.concat_tables([buy_rows, sell_rows, dealer_rows])

BUY_ROW_INDEX = 0
SELL_ROW_INDEX = 1
SELL_DEALER_ROW_INDEX = 2

def create_fmv_json(figi_row, generator: TraceDataGenerator, model: TraceModel, from_ordinals, quantity, execution_dates, report_dates):
    z_rows = create_fmv_z_rows(figi_row, from_ordinals, quantity,
                               execution_dates, report_dates)
    X_b, _ = generator.generate_batch(z_rows)
    Y_hat = model.evaluate_batch(X_b)
    if Y_hat.shape[0] != z_rows.shape[0]:
        raise ValueError("The evaluated Y_hat is not of the same length as the z rows.")
    model_prices = Y_hat[:, model.test_generator.get_rfq_label_index('price')]

    model_prices_json = []
    length = len(report_dates)
    for i in range(length):
        report_date = report_dates[i].as_py()
        buy_price = model_prices[BUY_ROW_INDEX * length + i]
        sell_price = model_prices[SELL_ROW_INDEX * length + i]
        dealer_price = model_prices[SELL_DEALER_ROW_INDEX * length + i]
        model_prices_json.append({
            "date": epoch_ns_to_iso_str(report_date),
            "prices": {"B": buy_price, "S": sell_price, "D": dealer_price}
        })
    return model_prices_json
