import settings
settings.override_if_main(__name__, 1)
from feed_forward_nn import FeedForwardNN
from generator import TraceDataGenerator
from trade_history import TradeHistory
from typical_trade import create_fmv_z_rows
from trace_data import get_trace_data
import pyarrow.compute as pc


if __name__ == '__main__':
    trace, ordinals = get_trace_data()
    print(trace.column_names)
    sequence_length = settings.get('$.data.trades.sequence_length')
    data_cube = TradeHistory(trace, ordinals, sequence_length)
    trace = data_cube.trace
    generator = TraceDataGenerator(data_cube, settings.get('$.batch_size'), should_shuffle=False)

    figi = 'BBG00JXGN2L3'
    figi_index = ordinals['figi'].index(figi)

    trace_figi = trace.filter(pc.equal(trace['figi'], figi_index))
    figi_row = trace_figi.slice(trace_figi.shape[0] - 1)
    fmv_query_rows = create_fmv_z_rows(figi_row, ordinals, None, None, None)

    model = FeedForwardNN(generator, generator, generator)
    model.create()
    X_b, Y_b = generator.generate_batch(fmv_query_rows)
    y_b_hat = model.evaluate_batch(X_b)

    print(fmv_query_rows.columns)
    print(fmv_query_rows)
    print(fmv_query_rows['execution_date'])
    print(fmv_query_rows['side'])
    print(y_b_hat)