from datetime import datetime, date
from typing import List, Dict

import pandas as pd
from peewee import (
    AutoField,
    CharField,
    DateTimeField,
    IntegerField,
    Model,
    MySQLDatabase as PeeweeMySQLDatabase,
    ModelSelect,
    ModelDelete,
    chunked,
    fn,
    DoubleField,
    DateField,
    BooleanField, BigIntegerField, ForeignKeyField, SQL, FloatField
)
from playhouse.shortcuts import ReconnectMixin
from werkzeug.security import generate_password_hash, check_password_hash

from vnpy.trader.constant import Exchange, Interval, Market, Conflict
from vnpy.trader.object import BarData, TickData
from vnpy.trader.database import (
    BaseDatabase,
    BarOverview,
    TickOverview,
    DB_TZ,
    convert_tz
)
from vnpy.trader.setting import SETTINGS
from ex_vnpy.object import BasicStockData, BasicIndexData, BasicSymbolData


class ReconnectMySQLDatabase(ReconnectMixin, PeeweeMySQLDatabase):
    """带有重连混入的MySQL数据库类"""
    pass


db = ReconnectMySQLDatabase(
    database=SETTINGS["database.database"],
    user=SETTINGS["database.user"],
    password=SETTINGS["database.password"],
    host=SETTINGS["database.host"],
    port=SETTINGS["database.port"]
)


class DateTimeMillisecondField(DateTimeField):
    """支持毫秒的日期时间戳字段"""

    def get_modifiers(self):
        """毫秒支持"""
        return [3]


class DbBarData(Model):
    """K线数据表映射对象"""

    id: AutoField = AutoField()

    symbol: str = CharField()
    exchange: str = CharField()
    datetime: datetime = DateTimeField()
    interval: str = CharField()

    volume: float = DoubleField()
    turnover: float = DoubleField()
    open_interest: float = DoubleField()
    open_price: float = DoubleField()
    high_price: float = DoubleField()
    low_price: float = DoubleField()
    close_price: float = DoubleField()

    class Meta:
        database: PeeweeMySQLDatabase = db
        indexes: tuple = ((("symbol", "exchange", "interval", "datetime"), True),)


class DbIndexBarData(Model):
    """指数的K线数据表映射对象"""

    id: AutoField = AutoField()

    symbol: str = CharField()
    exchange: str = CharField()
    datetime: datetime = DateTimeField()
    interval: str = CharField()

    volume: float = DoubleField()
    turnover: float = DoubleField()
    open_interest: float = DoubleField()
    open_price: float = DoubleField()
    high_price: float = DoubleField()
    low_price: float = DoubleField()
    close_price: float = DoubleField()

    class Meta:
        database: PeeweeMySQLDatabase = db
        indexes: tuple = ((("symbol", "exchange", "interval", "datetime"), True),)


class DbTickData(Model):
    """TICK数据表映射对象"""

    id: AutoField = AutoField()

    symbol: str = CharField()
    exchange: str = CharField()
    datetime: datetime = DateTimeMillisecondField()

    name: str = CharField()
    volume: float = DoubleField()
    turnover: float = DoubleField()
    open_interest: float = DoubleField()
    last_price: float = DoubleField()
    last_volume: float = DoubleField()
    limit_up: float = DoubleField()
    limit_down: float = DoubleField()

    open_price: float = DoubleField()
    high_price: float = DoubleField()
    low_price: float = DoubleField()
    pre_close: float = DoubleField()

    bid_price_1: float = DoubleField()
    bid_price_2: float = DoubleField(null=True)
    bid_price_3: float = DoubleField(null=True)
    bid_price_4: float = DoubleField(null=True)
    bid_price_5: float = DoubleField(null=True)

    ask_price_1: float = DoubleField()
    ask_price_2: float = DoubleField(null=True)
    ask_price_3: float = DoubleField(null=True)
    ask_price_4: float = DoubleField(null=True)
    ask_price_5: float = DoubleField(null=True)

    bid_volume_1: float = DoubleField()
    bid_volume_2: float = DoubleField(null=True)
    bid_volume_3: float = DoubleField(null=True)
    bid_volume_4: float = DoubleField(null=True)
    bid_volume_5: float = DoubleField(null=True)

    ask_volume_1: float = DoubleField()
    ask_volume_2: float = DoubleField(null=True)
    ask_volume_3: float = DoubleField(null=True)
    ask_volume_4: float = DoubleField(null=True)
    ask_volume_5: float = DoubleField(null=True)

    localtime: datetime = DateTimeMillisecondField(null=True)

    class Meta:
        database: PeeweeMySQLDatabase = db
        indexes: tuple = ((("symbol", "exchange", "datetime"), True),)


class DbBarOverview(Model):
    """K线汇总数据表映射对象"""

    id: AutoField = AutoField()

    symbol: str = CharField()
    exchange: str = CharField()
    interval: str = CharField()
    type: int = CharField()
    count: int = IntegerField()
    start: datetime = DateTimeField()
    end: datetime = DateTimeField()

    class Meta:
        database: PeeweeMySQLDatabase = db
        indexes: tuple = ((("symbol", "exchange", "interval", "type"), True),)


class DbTickOverview(Model):
    """Tick汇总数据表映射对象"""

    id: AutoField = AutoField()

    symbol: str = CharField()
    exchange: str = CharField()
    count: int = IntegerField()
    start: datetime = DateTimeField()
    end: datetime = DateTimeField()

    class Meta:
        database: PeeweeMySQLDatabase = db
        indexes: tuple = ((("symbol", "exchange"), True),)


class DbSymbol(Model):
    """股票、指数、基金等各投资品列表映射对象"""

    id = AutoField()
    symbol: str = CharField(null=False)
    name: str = CharField(null=False)
    exchange: str = CharField(null=False)
    market: str = CharField(null=False, default='CN')
    type: str = CharField(null=False, default='CS')
    status: str = CharField(null=False, default='active')
    update_dt: datetime = DateTimeField()

    class Meta:
        database: PeeweeMySQLDatabase = db
        indexes = ((("symbol", "exchange", "market", "type"), True),)


class DbStockMeta(Model):
    id = AutoField()

    symbol = ForeignKeyField(DbSymbol, backref='meta', verbose_name="股票基础信息")
    ex_date: date = DateField(default="'1970-01-01'")

    industry_first: str = CharField()
    industry_second: str = CharField()
    industry_third: str = CharField()
    industry_forth: str = CharField()

    industry_code_zz: str = CharField()
    industry_code: str = CharField(null=True)  # 统计局/证监会行业分类代码，可选

    index_sz50: bool = BooleanField(default=False)  # 默认为0
    index_hs300: bool = BooleanField(default=False)
    index_zz500: bool = BooleanField(default=False)
    index_zz800: bool = BooleanField(default=False)
    index_zz1000: bool = BooleanField(default=False)
    index_normal: bool = BooleanField(default=False)

    update_dt: datetime = DateTimeField()

    class Meta:
        database: PeeweeMySQLDatabase = db
        constraints = [SQL('UNIQUE(symbol_id)')]  # 添加对symbol的唯一性约束


class DbIndexMeta(Model):
    """Index列表映射对象"""

    id = AutoField()
    symbol = ForeignKeyField(DbSymbol, backref='meta', verbose_name="指数基础信息")

    full_name: str = CharField()

    volume: int = BigIntegerField(null=True)
    turnover: int = BigIntegerField(null=True)

    publish_date: date = DateField(null=False)
    exit_date: date = DateField(null=False)
    has_price: bool = BooleanField(default=True)
    has_weight: bool = BooleanField(default=True)
    has_components: bool = BooleanField(default=True)

    is_core_index: bool = BooleanField(default=False)

    class Meta:
        database: PeeweeMySQLDatabase = db
        constraints = [SQL('UNIQUE(symbol)')]  # 添加对symbol的唯一性约束


class DbBacktestingResults(Model):
    """股票回测结果
ALTER TABLE dbbacktestingresults add has_sl_HALow INT;
update dbbacktestingresults set has_sl_HALow = 0;
ALTER TABLE dbbacktestingresults MODIFY COLUMN has_sl_HALow INT NOT NULL;

ALTER TABLE dbbacktestingresults add do_sl_HALow INT;
update dbbacktestingresults set do_sl_HALow = 0;
ALTER TABLE dbbacktestingresults MODIFY COLUMN do_sl_HALow INT NOT NULL;
    """

    id = AutoField()
    symbol: str = CharField()
    name: str = CharField()
    exchange: str = CharField()
    market: str = CharField()

    backtesting_dt: date = DateField()
    update_dt: datetime = DateTimeField()

    start_date: datetime = DateTimeField()
    end_date: datetime = DateTimeField()
    total_days: int = IntegerField()
    profit_days: int = IntegerField()
    loss_days: int = IntegerField()
    capital: float = DoubleField()
    end_balance: float = DoubleField()
    max_drawdown: float = DoubleField()
    max_ddpercent: float = DoubleField()
    max_drawdown_duration: int = IntegerField()
    total_net_pnl: float = DoubleField()
    daily_net_pnl: float = DoubleField()
    total_commission: float = DoubleField()
    daily_commission: float = DoubleField()
    total_slippage: float = DoubleField()
    daily_slippage: float = DoubleField()
    total_turnover: float = DoubleField()
    daily_turnover: float = DoubleField()
    total_trade_count: int = IntegerField()
    daily_trade_count: float = DoubleField()
    total_return: float = DoubleField()
    annual_return: float = DoubleField()
    daily_return: float = DoubleField()
    return_std: float = DoubleField()
    sharpe_ratio: float = DoubleField()
    return_drawdown_ratio: float = DoubleField()
    win_rate_weighted: float = DoubleField()
    loss_rate_weighted: float = DoubleField()
    win_rate_normal: float = DoubleField()
    total_entry_count: int = IntegerField()
    entry_win_count: int = IntegerField()
    entry_loss_count: int = IntegerField()
    win_count_8: int = IntegerField()
    win_count_16: int = IntegerField()
    win_count_16a: int = IntegerField()
    loss_count_2: int = IntegerField()
    loss_count_5: int = IntegerField()
    loss_count_8: int = IntegerField()
    loss_count_8a: int = IntegerField()
    total_has_sl_count: int = IntegerField()
    total_do_sl_count: int = IntegerField()
    has_sl_Empty: int = IntegerField()
    has_sl_Init: int = IntegerField()
    has_sl_Detector: int = IntegerField()
    has_sl_Dynamic: int = IntegerField()
    has_sl_LowTwo: int = IntegerField()
    has_sl_LowFive: int = IntegerField()
    has_sl_EnterTwo: int = IntegerField()
    has_sl_Impulse: int = IntegerField()
    has_sl_Ema: int = IntegerField()
    has_sl_LargeUp: int = IntegerField()
    has_sl_LargeVolume: int = IntegerField()
    has_sl_Engine: int = IntegerField()
    has_sl_LostSpeed: int = IntegerField()
    has_sl_LevelLargeUp: int = IntegerField()
    has_sl_LargeDrop: int = IntegerField()
    has_sl_LostMovement: int = IntegerField()
    has_sl_TopPivot: int = IntegerField()
    has_sl_HALow: int = IntegerField()
    do_sl_Init: int = IntegerField()
    do_sl_Detector: int = IntegerField()
    do_sl_Dynamic: int = IntegerField()
    do_sl_LowTwo: int = IntegerField()
    do_sl_LowFive: int = IntegerField()
    do_sl_EnterTwo: int = IntegerField()
    do_sl_Impulse: int = IntegerField()
    do_sl_Ema: int = IntegerField()
    do_sl_LargeUp: int = IntegerField()
    do_sl_LargeVolume: int = IntegerField()
    do_sl_Engine: int = IntegerField()
    do_sl_LostSpeed: int = IntegerField()
    do_sl_LevelLargeUp: int = IntegerField()
    do_sl_LargeDrop: int = IntegerField()
    do_sl_LostMovement: int = IntegerField()
    do_sl_TopPivot: int = IntegerField()
    do_sl_HALow: int = IntegerField()

    class Meta:
        database: PeeweeMySQLDatabase = db
        indexes = ((("symbol", "exchange", "market", "backtesting_dt"), True),)

    def save(self, *args, **kwargs):
        self.update_dt = datetime.now()
        return super(DbBacktestingResults, self).save(*args, **kwargs)


class DbStockCapitalData(Model):
    """股票列表映射对象"""

    id = AutoField()

    symbol: str = CharField(max_length=32)
    date: date = DateField()
    level: str = CharField(max_length=32)
    role: str = CharField(max_length=16)
    trade_time: str = CharField(max_length=8)
    direction: str = CharField(max_length=8)
    interval: str = CharField(max_length=8)
    type: str = CharField(max_length=8)
    order_count: int = IntegerField()
    order_volume: int = IntegerField()
    turnover: float = DoubleField()
    volume: int = IntegerField()

    class Meta:
        database: PeeweeMySQLDatabase = db
        indexes = ((("symbol", "date", "trade_time", "level", "direction", "interval"), True),)

class DbStockCapitalDataNew(Model):
    """股票列表映射对象"""

    id = AutoField()

    symbol: str = CharField(max_length=32)
    symbol_meta = ForeignKeyField(DbSymbol, column_name='symbol_id', backref='capital', verbose_name="股票基础信息")
    # symbol_id: int = IntegerField(null=True)
    date: date = DateField()
    level: str = CharField(max_length=2)
    direction: str = CharField(max_length=1)
    trade_time: str = CharField(max_length=4)
    interval: str = CharField(max_length=2)
    role: str = CharField(max_length=1)
    order_count: int = IntegerField()
    order_volume: int = IntegerField()
    turnover: float = FloatField()
    volume: int = IntegerField()

    class Meta:
        database: PeeweeMySQLDatabase = db
        indexes = ((("symbol", "date", "trade_time", "level", "direction", "interval"), True),)


class DbStockCapitalFlatData(Model):
    id = AutoField()

    symbol: str = CharField(max_length=32)
    date: date = DateField()
    interval: str = CharField(max_length=8)
    type: str = CharField(max_length=8)
    order_count_buy_XL: int = IntegerField()
    order_count_buy_L: int = IntegerField()
    order_count_buy_M: int = IntegerField()
    order_count_buy_S: int = IntegerField()
    order_count_sale_XL: int = IntegerField()
    order_count_sale_L: int = IntegerField()
    order_count_sale_M: int = IntegerField()
    order_count_sale_S: int = IntegerField()
    order_volume_buy_XL: int = IntegerField()
    order_volume_buy_L: int = IntegerField()
    order_volume_buy_M: int = IntegerField()
    order_volume_buy_S: int = IntegerField()
    order_volume_sale_XL: int = IntegerField()
    order_volume_sale_L: int = IntegerField()
    order_volume_sale_M: int = IntegerField()
    order_volume_sale_S: int = IntegerField()
    turnover_buy_XL: int = IntegerField()
    turnover_buy_L: int = IntegerField()
    turnover_buy_M: int = IntegerField()
    turnover_buy_S: int = IntegerField()
    turnover_sale_XL: int = IntegerField()
    turnover_sale_L: int = IntegerField()
    turnover_sale_M: int = IntegerField()
    turnover_sale_S: int = IntegerField()
    volume_buy_XL: int = IntegerField()
    volume_buy_L: int = IntegerField()
    volume_buy_M: int = IntegerField()
    volume_buy_S: int = IntegerField()
    volume_sale_XL: int = IntegerField()
    volume_sale_L: int = IntegerField()
    volume_sale_M: int = IntegerField()
    volume_sale_S: int = IntegerField()

    class Meta:
        database: PeeweeMySQLDatabase = db
        indexes = ((("symbol", "date", "interval", "type"), True),)

class DbStockCapitalFlatDataNew(Model):
    id = AutoField()

    symbol: str = CharField(max_length=32)
    symbol_meta = ForeignKeyField(DbSymbol, column_name='symbol_id', backref='capital_flat', verbose_name="股票基础信息")
    # symbol_id: int = IntegerField(null=True)
    datetime: datetime = DateTimeField()
    interval: str = CharField(max_length=8)
    order_count_buy_XL: int = IntegerField()
    order_count_buy_L: int = IntegerField()
    order_count_buy_M: int = IntegerField()
    order_count_buy_S: int = IntegerField()
    order_count_sell_XL: int = IntegerField()
    order_count_sell_L: int = IntegerField()
    order_count_sell_M: int = IntegerField()
    order_count_sell_S: int = IntegerField()
    order_volume_buy_XL: int = IntegerField()
    order_volume_buy_L: int = IntegerField()
    order_volume_buy_M: int = IntegerField()
    order_volume_buy_S: int = IntegerField()
    order_volume_sell_XL: int = IntegerField()
    order_volume_sell_L: int = IntegerField()
    order_volume_sell_M: int = IntegerField()
    order_volume_sell_S: int = IntegerField()
    turnover_buy_XL: int = IntegerField()
    turnover_buy_L: int = IntegerField()
    turnover_buy_M: int = IntegerField()
    turnover_buy_S: int = IntegerField()
    turnover_sell_XL: int = IntegerField()
    turnover_sell_L: int = IntegerField()
    turnover_sell_M: int = IntegerField()
    turnover_sell_S: int = IntegerField()
    volume_buy_XL: int = IntegerField()
    volume_buy_L: int = IntegerField()
    volume_buy_M: int = IntegerField()
    volume_buy_S: int = IntegerField()
    volume_sell_XL: int = IntegerField()
    volume_sell_L: int = IntegerField()
    volume_sell_M: int = IntegerField()
    volume_sell_S: int = IntegerField()

    class Meta:
        database: PeeweeMySQLDatabase = db
        indexes = ((("symbol", "datetime", "interval"), True),)

class DbCapitalOverview(Model):
    """Level2汇总数据表映射对象"""

    id: AutoField = AutoField()

    symbol = ForeignKeyField(DbSymbol, backref='capital_overview', verbose_name="股票基础信息")
    interval: str = CharField()
    count: int = IntegerField()
    start: date = DateField()
    end: date = DateField()

    update_dt: datetime = DateTimeField()

    class Meta:
        database: PeeweeMySQLDatabase = db
        indexes = ((("symbol_id", "interval"), True),)


class DbUserData(Model):
    """股票列表映射对象"""

    id = AutoField()
    email: str = CharField(max_length=32, unique=True)
    username: str = CharField(max_length=32)
    password_hash: str = CharField(max_length=256)

    update_dt: datetime = DateTimeField()

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def save(self, *args, **kwargs):
        self.update_dt = datetime.now()
        return super(DbUserData, self).save(*args, **kwargs)

    class Meta:
        database: PeeweeMySQLDatabase = db
        # indexes = (("email", True),)


class DbSymbolLists(Model):
    """股票列表映射对象"""

    id = AutoField()
    name: str = CharField(max_length=128, null=False)
    # rank: 表示排行榜，比如每日的龙虎榜
    # board: 交易所板块
    # index: 表示指数列表
    # industry: 表示行业板块
    # concept: 表示概念板块
    # area: 表示地域板块
    # personal: 表示个人自选股
    # system: 表示系统用途的列表
    type: str = CharField(max_length=32, null=False)
    dt: date = DateField()
    create_dt: datetime = DateTimeField()
    update_dt: datetime = DateTimeField()

    class Meta:
        database: PeeweeMySQLDatabase = db
        indexes = ((("name", 'type', 'dt'), True),)


class DbSymbolListMap(Model):
    """股票列表关系对象"""

    id = AutoField()
    symbol_list = ForeignKeyField(DbSymbolLists, backref='symbols', verbose_name="股票列表")
    symbol = ForeignKeyField(DbSymbol, backref='lists', verbose_name="股票")
    create_dt: datetime = DateTimeField()
    update_dt: datetime = DateTimeField()

    class Meta:
        database: PeeweeMySQLDatabase = db
        indexes = (
            # 创建一个联合索引
            (('symbol_list', 'symbol'), True),
        )


class DbDailyStatData(Model):
    id = AutoField()
    symbol = ForeignKeyField(DbSymbol, backref='meta', verbose_name="股票基础信息")
    datetime: datetime = DateTimeField()
    interval: str = CharField()

    # last_price: float = DoubleField()
    change_pct: float = DoubleField()
    volume_ratio: float = DoubleField()
    # large_capital_amount_in: float = DoubleField()      # 主力资金流入
    change_pct_5u: float = DoubleField()
    change_pct_10u: float = DoubleField()
    change_pct_22u: float = DoubleField()

    # 以下仅适用于interval=d
    cont_up_days: float = IntegerField()  # 连续上涨天数
    cont_max_up_days: float = IntegerField()  # 连续涨停天数，最大连续间断不超过1天

    update_dt: datetime = DateTimeField()

    class Meta:
        database: PeeweeMySQLDatabase = db
        indexes: tuple = ((("symbol", "interval", "datetime"), True),)


# a new model to log operation
class DbOperation(Model):
    id = AutoField()
    op_type: str = CharField(max_length=128, null=False)
    op_status: str = CharField(max_length=128, null=False)
    op_time: datetime = DateTimeField()
    op_info: str = CharField(max_length=128)
    update_dt: datetime = DateTimeField()

    class Meta:
        database: PeeweeMySQLDatabase = db


class DbAliyunBinlogFiles(Model):
    id = AutoField()

    binlog_file: str = CharField(max_length=128, null=False)
    file_size: int = IntegerField()
    remote_status: str = CharField(max_length=128, null=False, default='Completed')
    local_status: str = CharField(max_length=128, null=False, default='New')
    download_link: str = CharField(max_length=2048, null=False)
    log_start_time: datetime = DateTimeField()
    log_end_time: datetime = DateTimeField()
    link_expire_time: datetime = DateTimeField()
    update_dt: datetime = DateTimeField()

    class Meta:
        database: PeeweeMySQLDatabase = db
        indexes = ((("binlog_file", "remote_status"), True),)


class MysqlDatabase(BaseDatabase):
    """Mysql数据库接口"""

    def __init__(self) -> None:
        """"""
        self.db: PeeweeMySQLDatabase = db
        self.db.connect()
        self.db.create_tables([DbBarData, DbTickData, DbBarOverview, DbTickOverview])
        self.db.create_tables([DbSymbol, DbStockMeta, DbIndexMeta, DbSymbolLists, DbSymbolListMap])
        self.db.create_tables([DbStockCapitalDataNew, DbStockCapitalFlatDataNew])
        self.db.create_tables([DbUserData, DbOperation, DbAliyunBinlogFiles, DbBacktestingResults])

    def save_index_bar_data(self, bars: List[BarData], stream: bool = False,
                            conflict: Conflict = Conflict.REPLACE) -> bool:
        """保存Index K线数据"""
        # 读取主键参数
        bar: BarData = bars[0]
        symbol: str = bar.symbol
        exchange: Exchange = bar.exchange
        interval: Interval = bar.interval

        # 将BarData数据转换为字典，并调整时区
        data: list = []

        for bar in bars:
            bar.datetime = convert_tz(bar.datetime)

            d: dict = bar.__dict__
            d["exchange"] = d["exchange"].value
            d["interval"] = d["interval"].value
            d.pop("gateway_name")
            d.pop("vt_symbol")
            data.append(d)

        # 使用upsert操作将数据更新到数据库中
        with self.db.atomic():
            if conflict == Conflict.IGNORE:
                for c in chunked(data, 50):
                    DbIndexBarData.insert_many(c).on_conflict_ignore().execute()
            else:
                for c in chunked(data, 50):
                    DbIndexBarData.insert_many(c).on_conflict_replace().execute()

        # 更新K线汇总数据
        overview: DbBarOverview = DbBarOverview.get_or_none(
            DbBarOverview.symbol == symbol,
            DbBarOverview.exchange == exchange.value,
            DbBarOverview.interval == interval.value,
            DbBarOverview.type == "INDX",
        )

        if not overview:
            overview: DbBarOverview = DbBarOverview()
            overview.symbol = symbol
            overview.exchange = exchange.value
            overview.interval = interval.value
            overview.start = bars[0].datetime
            overview.end = bars[-1].datetime
            overview.type = "INDX"
            overview.count = len(bars)
        elif stream:
            overview.end = bars[-1].datetime
            overview.count += len(bars)
        else:
            overview.start = min(bars[0].datetime, overview.start)
            overview.end = max(bars[-1].datetime, overview.end)

            s: ModelSelect = DbIndexBarData.select().where(
                (DbIndexBarData.symbol == symbol)
                & (DbIndexBarData.exchange == exchange.value)
                & (DbIndexBarData.interval == interval.value)
            )
            overview.count = s.count()

        overview.save()

        return True

    def save_bar_data(self, bars: List[BarData], stream: bool = False, conflict: Conflict = Conflict.REPLACE) -> bool:
        """保存K线数据"""
        # 读取主键参数
        bar: BarData = bars[0]
        symbol: str = bar.symbol
        exchange: Exchange = bar.exchange
        interval: Interval = bar.interval

        # 将BarData数据转换为字典，并调整时区
        data: list = []

        for bar in bars:
            bar.datetime = convert_tz(bar.datetime)

            d: dict = bar.__dict__
            d["exchange"] = d["exchange"].value
            d["interval"] = d["interval"].value
            d.pop("gateway_name")
            d.pop("vt_symbol")
            data.append(d)

        # 使用upsert操作将数据更新到数据库中
        with self.db.atomic():
            if conflict == Conflict.IGNORE:
                for c in chunked(data, 50):
                    DbBarData.insert_many(c).on_conflict_ignore().execute()
            else:
                for c in chunked(data, 50):
                    DbBarData.insert_many(c).on_conflict_replace().execute()

        # 更新K线汇总数据
        overview: DbBarOverview = DbBarOverview.get_or_none(
            DbBarOverview.symbol == symbol,
            DbBarOverview.exchange == exchange.value,
            DbBarOverview.interval == interval.value,
            DbBarOverview.type == "CS",
        )

        if not overview:
            overview: DbBarOverview = DbBarOverview()
            overview.symbol = symbol
            overview.exchange = exchange.value
            overview.interval = interval.value
            overview.start = bars[0].datetime
            overview.end = bars[-1].datetime
            overview.type = "CS"
            overview.count = len(bars)
        elif stream:
            overview.end = bars[-1].datetime
            overview.count += len(bars)
        else:
            overview.start = min(bars[0].datetime, overview.start)
            overview.end = max(bars[-1].datetime, overview.end)

            s: ModelSelect = DbBarData.select().where(
                (DbBarData.symbol == symbol)
                & (DbBarData.exchange == exchange.value)
                & (DbBarData.interval == interval.value)
            )
            overview.count = s.count()

        overview.save()

        return True

    def save_tick_data(self, ticks: List[TickData], stream: bool = False,
                       conflict: Conflict = Conflict.REPLACE) -> bool:
        """保存TICK数据"""
        # 读取主键参数
        tick: TickData = ticks[0]
        symbol: str = tick.symbol
        exchange: Exchange = tick.exchange

        # 将TickData数据转换为字典，并调整时区
        data: list = []

        for tick in ticks:
            tick.datetime = convert_tz(tick.datetime)

            d: dict = tick.__dict__
            d["exchange"] = d["exchange"].value
            d.pop("gateway_name")
            d.pop("vt_symbol")
            data.append(d)

        # 使用upsert操作将数据更新到数据库中
        with self.db.atomic():
            if conflict == Conflict.IGNORE:
                for c in chunked(data, 50):
                    DbTickData.insert_many(c).on_conflict_ignore().execute()
            else:
                for c in chunked(data, 50):
                    DbTickData.insert_many(c).on_conflict_replace().execute()

        # 更新Tick汇总数据
        overview: DbTickOverview = DbTickOverview.get_or_none(
            DbTickOverview.symbol == symbol,
            DbTickOverview.exchange == exchange.value,
        )

        if not overview:
            overview: DbTickOverview = DbTickOverview()
            overview.symbol = symbol
            overview.exchange = exchange.value
            overview.start = ticks[0].datetime
            overview.end = ticks[-1].datetime
            overview.count = len(ticks)
        elif stream:
            overview.end = ticks[-1].datetime
            overview.count += len(ticks)
        else:
            overview.start = min(ticks[0].datetime, overview.start)
            overview.end = max(ticks[-1].datetime, overview.end)

            s: ModelSelect = DbTickData.select().where(
                (DbTickData.symbol == symbol)
                & (DbTickData.exchange == exchange.value)
            )
            overview.count = s.count()

        overview.save()

        return True

    def load_bar_data(
            self,
            symbol: str,
            exchange: Exchange,
            interval: Interval,
            start: datetime,
            end: datetime
    ) -> List[BarData]:
        """"""
        s: ModelSelect = (
            DbBarData.select().where(
                (DbBarData.symbol == symbol)
                & (DbBarData.exchange == exchange.value)
                & (DbBarData.interval == interval.value)
                & (DbBarData.datetime >= start)
                & (DbBarData.datetime <= end)
            ).order_by(DbBarData.datetime)
        )

        bars: List[BarData] = []
        for db_bar in s:
            bar: BarData = BarData(
                symbol=db_bar.symbol,
                exchange=Exchange(db_bar.exchange),
                datetime=datetime.fromtimestamp(db_bar.datetime.timestamp(), DB_TZ),
                interval=Interval(db_bar.interval),
                volume=db_bar.volume,
                turnover=db_bar.turnover,
                open_interest=db_bar.open_interest,
                open_price=db_bar.open_price,
                high_price=db_bar.high_price,
                low_price=db_bar.low_price,
                close_price=db_bar.close_price,
                gateway_name="DB"
            )
            bars.append(bar)

        return bars

    def load_index_bar_data(
            self,
            symbol: str,
            exchange: Exchange,
            interval: Interval,
            start: datetime,
            end: datetime
    ) -> List[BarData]:
        """"""
        s: ModelSelect = (
            DbIndexBarData.select().where(
                (DbIndexBarData.symbol == symbol)
                & (DbIndexBarData.exchange == exchange.value)
                & (DbIndexBarData.interval == interval.value)
                & (DbIndexBarData.datetime >= start)
                & (DbIndexBarData.datetime <= end)
            ).order_by(DbIndexBarData.datetime)
        )

        bars: List[BarData] = []
        for db_bar in s:
            bar: BarData = BarData(
                symbol=db_bar.symbol,
                exchange=Exchange(db_bar.exchange),
                datetime=datetime.fromtimestamp(db_bar.datetime.timestamp(), DB_TZ),
                interval=Interval(db_bar.interval),
                volume=db_bar.volume,
                turnover=db_bar.turnover,
                open_interest=db_bar.open_interest,
                open_price=db_bar.open_price,
                high_price=db_bar.high_price,
                low_price=db_bar.low_price,
                close_price=db_bar.close_price,
                gateway_name="DB"
            )
            bars.append(bar)

        return bars

    def load_tick_data(
            self,
            symbol: str,
            exchange: Exchange,
            start: datetime,
            end: datetime
    ) -> List[TickData]:
        """读取TICK数据"""
        s: ModelSelect = (
            DbTickData.select().where(
                (DbTickData.symbol == symbol)
                & (DbTickData.exchange == exchange.value)
                & (DbTickData.datetime >= start)
                & (DbTickData.datetime <= end)
            ).order_by(DbTickData.datetime)
        )

        ticks: List[TickData] = []
        for db_tick in s:
            tick: TickData = TickData(
                symbol=db_tick.symbol,
                exchange=Exchange(db_tick.exchange),
                datetime=datetime.fromtimestamp(db_tick.datetime.timestamp(), DB_TZ),
                name=db_tick.o_name,
                volume=db_tick.volume,
                turnover=db_tick.turnover,
                open_interest=db_tick.open_interest,
                last_price=db_tick.last_price,
                last_volume=db_tick.last_volume,
                limit_up=db_tick.limit_up,
                limit_down=db_tick.limit_down,
                open_price=db_tick.open_price,
                high_price=db_tick.high_price,
                low_price=db_tick.low_price,
                pre_close=db_tick.pre_close,
                bid_price_1=db_tick.bid_price_1,
                bid_price_2=db_tick.bid_price_2,
                bid_price_3=db_tick.bid_price_3,
                bid_price_4=db_tick.bid_price_4,
                bid_price_5=db_tick.bid_price_5,
                ask_price_1=db_tick.ask_price_1,
                ask_price_2=db_tick.ask_price_2,
                ask_price_3=db_tick.ask_price_3,
                ask_price_4=db_tick.ask_price_4,
                ask_price_5=db_tick.ask_price_5,
                bid_volume_1=db_tick.bid_volume_1,
                bid_volume_2=db_tick.bid_volume_2,
                bid_volume_3=db_tick.bid_volume_3,
                bid_volume_4=db_tick.bid_volume_4,
                bid_volume_5=db_tick.bid_volume_5,
                ask_volume_1=db_tick.ask_volume_1,
                ask_volume_2=db_tick.ask_volume_2,
                ask_volume_3=db_tick.ask_volume_3,
                ask_volume_4=db_tick.ask_volume_4,
                ask_volume_5=db_tick.ask_volume_5,
                localtime=db_tick.localtime,
                gateway_name="DB"
            )
            ticks.append(tick)

        return ticks

    def delete_bar_data(
            self,
            symbol: str,
            exchange: Exchange,
            interval: Interval
    ) -> int:
        """删除K线数据"""
        d: ModelDelete = DbBarData.delete().where(
            (DbBarData.symbol == symbol)
            & (DbBarData.exchange == exchange.value)
            & (DbBarData.interval == interval.value)
        )
        count: int = d.execute()

        # 删除K线汇总数据
        d2: ModelDelete = DbBarOverview.delete().where(
            (DbBarOverview.symbol == symbol)
            & (DbBarOverview.exchange == exchange.value)
            & (DbBarOverview.interval == interval.value)
        )
        d2.execute()
        return count

    def delete_tick_data(
            self,
            symbol: str,
            exchange: Exchange
    ) -> int:
        """删除TICK数据"""
        d: ModelDelete = DbTickData.delete().where(
            (DbTickData.symbol == symbol)
            & (DbTickData.exchange == exchange.value)
        )

        count: int = d.execute()

        # 删除Tick汇总数据
        d2: ModelDelete = DbTickOverview.delete().where(
            (DbTickOverview.symbol == symbol)
            & (DbTickOverview.exchange == exchange.value)
        )
        d2.execute()
        return count

    def get_bar_overview(self, symbol_type: str = "CS") -> List[BarOverview]:
        """查询数据库中的K线汇总信息"""
        # 如果已有K线，但缺失汇总信息，则执行初始化
        data_count: int = DbBarData.select().count() if symbol_type == "CS" else DbIndexBarData.select().count()
        overview_count: int = DbBarOverview.select().where((DbBarOverview.type == symbol_type)).count()
        if data_count and not overview_count:
            if symbol_type == "CS":
                self.init_bar_overview()
            else:
                self.init_index_bar_overview()

        s: ModelSelect = DbBarOverview.select().where((DbBarOverview.type == symbol_type))
        overviews: List[BarOverview] = []
        for overview in s:
            overview.exchange = Exchange(overview.exchange)
            overview.interval = Interval(overview.interval)
            overview.start = datetime.fromtimestamp(overview.start.timestamp(), DB_TZ)
            overview.end = datetime.fromtimestamp(overview.end.timestamp(), DB_TZ)
            overviews.append(overview)
        return overviews

    def get_tick_overview(self) -> List[TickOverview]:
        """查询数据库中的Tick汇总信息"""
        s: ModelSelect = DbTickOverview.select()
        overviews: list = []
        for overview in s:
            overview.exchange = Exchange(overview.exchange)
            overviews.append(overview)
        return overviews

    def init_bar_overview(self) -> None:
        """初始化数据库中的K线汇总信息"""
        s: ModelSelect = (
            DbBarData.select(
                DbBarData.symbol,
                DbBarData.exchange,
                DbBarData.interval,
                fn.COUNT(DbBarData.id).alias("count")
            ).group_by(
                DbBarData.symbol,
                DbBarData.exchange,
                DbBarData.interval
            )
        )

        for data in s:
            overview: DbBarOverview = DbBarOverview()
            overview.symbol = data.symbol
            overview.exchange = data.exchange
            overview.interval = data.interval
            overview.type = "CS"
            overview.count = data.count

            start_bar: DbBarData = (
                DbBarData.select()
                .where(
                    (DbBarData.symbol == data.symbol)
                    & (DbBarData.exchange == data.exchange)
                    & (DbBarData.interval == data.interval)
                )
                .order_by(DbBarData.datetime.asc())
                .first()
            )
            overview.start = start_bar.datetime

            end_bar: DbBarData = (
                DbBarData.select()
                .where(
                    (DbBarData.symbol == data.symbol)
                    & (DbBarData.exchange == data.exchange)
                    & (DbBarData.interval == data.interval)
                )
                .order_by(DbBarData.datetime.desc())
                .first()
            )
            overview.end = end_bar.datetime

            overview.save()

    def init_index_bar_overview(self) -> None:
        """初始化数据库中的K线汇总信息"""
        s: ModelSelect = (
            DbIndexBarData.select(
                DbIndexBarData.symbol,
                DbIndexBarData.exchange,
                DbIndexBarData.interval,
                fn.COUNT(DbIndexBarData.id).alias("count")
            ).group_by(
                DbIndexBarData.symbol,
                DbIndexBarData.exchange,
                DbIndexBarData.interval
            )
        )

        for data in s:
            overview: DbBarOverview = DbBarOverview()
            overview.symbol = data.symbol
            overview.exchange = data.exchange
            overview.interval = data.interval
            overview.type = "INDX"
            overview.count = data.count

            start_bar: DbIndexBarData = (
                DbIndexBarData.select()
                .where(
                    (DbIndexBarData.symbol == data.symbol)
                    & (DbIndexBarData.exchange == data.exchange)
                    & (DbIndexBarData.interval == data.interval)
                )
                .order_by(DbIndexBarData.datetime.asc())
                .first()
            )
            overview.start = start_bar.datetime

            end_bar: DbIndexBarData = (
                DbIndexBarData.select()
                .where(
                    (DbIndexBarData.symbol == data.symbol)
                    & (DbIndexBarData.exchange == data.exchange)
                    & (DbIndexBarData.interval == data.interval)
                )
                .order_by(DbIndexBarData.datetime.desc())
                .first()
            )
            overview.end = end_bar.datetime

            overview.save()

    def get_symbol_ids(self, s_type: str, market: Market) -> Dict[str, int]:
        s = DbSymbol.select(DbSymbol.id, DbSymbol.symbol).where(
            (DbSymbol.type == s_type) & (DbSymbol.market == market.value)).dicts()
        id_maps = {}
        for d in s:
            id_maps[d["symbol"]] = d["id"]
        return id_maps

    def get_basic_stock_data(self) -> Dict[Market, List[BasicStockData]]:
        """查询数据库中的基础信息汇总信息"""

        s = DbSymbol.select(DbSymbol, DbStockMeta).join(DbStockMeta).where(
            (DbSymbol.status == 'active') &
            (DbSymbol.type == 'CS')
        ).dicts()
        overviews = {}
        for ov in s:
            ov["exchange"] = Exchange(ov['exchange'])
            market = Market(ov['market'])
            ov['market'] = market
            if market not in overviews:
                overviews[market] = []
            overviews[market].append(BasicStockData(**ov))
        return overviews

    def get_basic_index_data(self) -> Dict[Market, List[BasicIndexData]]:
        """查询数据库中的基础信息汇总信息"""

        s = DbSymbol.select(DbSymbol, DbIndexMeta).join(DbIndexMeta).where(
            (DbSymbol.status == 'active') &
            (DbSymbol.type == 'INDX')
        ).dicts()
        overviews = {}
        for ov in s:
            ov["exchange"] = Exchange(ov['exchange'])
            market = Market(ov['market'])
            ov['market'] = market
            if market not in overviews:
                overviews[market] = []
            overviews[market].append(BasicIndexData(**ov))
        return overviews

    def get_basic_info_by_symbols(self, symbols, market: Market = Market.CN, symbol_type: str = 'CS') -> List[
        BasicSymbolData]:
        """查询数据库中的基础信息"""

        basic_datas: List[BasicSymbolData] = []

        if symbol_type == 'CS':
            stocks = (
                DbSymbol.select(DbSymbol, DbStockMeta)
                .join(DbStockMeta)
                .where(
                    (DbSymbol.status == 'active') &
                    (DbSymbol.market == market.value) &
                    (DbSymbol.type == symbol_type) &
                    (DbSymbol.symbol.in_(symbols)))
                .dicts()
            )

            for dc in stocks:
                dc["exchange"] = Exchange(dc['exchange'])
                dc["market"] = Market(dc['market'])
                basic_datas.append(BasicStockData(**dc))

        elif symbol_type == 'INDX':
            indexes = (
                DbSymbol.select(DbSymbol, DbIndexMeta)
                .join(DbIndexMeta)
                .where(
                    (DbSymbol.status == 'active') &
                    (DbSymbol.market == market.value) &
                    (DbSymbol.type == symbol_type) &
                    (DbSymbol.symbol.in_(symbols)))
                .dicts()
            )
            for dc in indexes:
                dc["exchange"] = Exchange(dc['exchange'])
                dc["market"] = Market(dc['market'])
                basic_datas.append(BasicIndexData(**dc))

        return basic_datas

    def update_daily_stat_data(self, many_data: List, conflict: Conflict = Conflict.IGNORE):
        """更新每日统计数据"""
        with self.db.atomic():
            if conflict == Conflict.IGNORE:
                for c in chunked(many_data, 50):
                    DbDailyStatData.insert_many(c).on_conflict_ignore().execute()
            else:
                for c in chunked(many_data, 50):
                    DbDailyStatData.insert_many(c).on_conflict_replace().execute()

    def save_operation_log(self, op_type: str, op_status: str, op_time: datetime, op_info: str = ""):
        # 插入新数据
        operation = DbOperation.create(
            op_type=op_type,
            op_status=op_status,
            op_time=op_time,
            op_info=op_info,
            update_dt=datetime.now(DB_TZ)
        )

        # 保存到数据库
        operation.save()

    def save_capital_data(self, data: List):
        with self.db.atomic():
            for c in chunked(data, 1000):
                DbStockCapitalDataNew.insert_many(c).on_conflict_replace().execute()

    def save_capital_flat_data(self, data: List):
        with self.db.atomic():
            for c in chunked(data, 1000):
                DbStockCapitalFlatDataNew.insert_many(c).on_conflict_replace().execute()

    def update_stocks_meta_data(self, stocks_df, market: Market):
        # 第一部分：更新或插入数据
        symbols_dict = stocks_df[['symbol', 'name', 'exchange', 'market', 'type', 'status', 'update_dt']].to_dict(
            'records')
        DbSymbol.insert_many(symbols_dict).on_conflict(
            preserve=[DbSymbol.name, DbSymbol.status, DbSymbol.update_dt]
        ).execute()

        symbols = stocks_df['symbol'].to_list()

        # 查询这些symbol的当前状态，包括它们的ID
        db_symbols = (DbSymbol.select(DbSymbol.id, DbSymbol.symbol)
                      .where((DbSymbol.symbol.in_(symbols)) &
                             (DbSymbol.type == 'CS') &
                             (DbSymbol.market == market.value)
                             )
                      .dicts())
        # convert db_symbols to dataframe and merge stocks_df
        db_symbols_df = pd.DataFrame(db_symbols)
        stocks_data = pd.merge(stocks_df, db_symbols_df, on='symbol', how='left')
        meta_df = stocks_data.drop(
            ['symbol', 'name', 'exchange', 'market', 'type', 'status', 'shares_circ_a', 'shares_non_circ_a',
             'shares_total_a', 'shares_total'], axis=1)
        meta_dict = meta_df.rename(columns={'id': 'symbol_id'}).to_dict('records')

        preserve_fields = [field for field in DbStockMeta._meta.fields.keys() if
                           field not in ['id', 'symbol', 'symbol_id']]
        DbStockMeta.insert_many(meta_dict).on_conflict(preserve=preserve_fields).execute()

        # 第二部分：更新未包含在列表中的DbSymbol的状态
        query = DbSymbol.update(status='inactive').where(
            (DbSymbol.symbol.not_in(symbols)) &
            (DbSymbol.market == market.value) &
            (DbSymbol.type == 'CS')
        )
        query.execute()

    def get_capital_days(self, start_date: datetime, end_date: datetime) -> List[str]:
        # 查询并去重
        query = (DbStockCapitalFlatDataNew
                 .select(DbStockCapitalFlatDataNew.datetime)
                 .where((DbStockCapitalFlatDataNew.datetime >= start_date) &
                        (DbStockCapitalFlatDataNew.datetime < end_date) &
                        (DbStockCapitalFlatDataNew.interval == 'd'))
                 .group_by(DbStockCapitalFlatDataNew.datetime)
                 .order_by(DbStockCapitalFlatDataNew.datetime))

        # 提取日部分，并格式化为 dd 格式
        days_dd = [record.datetime.strftime('%d') for record in query]

        return days_dd

    def get_latest_statistic_date(self):
        latest = (DbDailyStatData.select()
                     .where(DbDailyStatData.interval == 'd')
                     .order_by(DbDailyStatData.datetime.desc())
                     .first())
        return datetime.fromtimestamp(latest.datetime.timestamp(), DB_TZ)

    def get_latest_op_info(self, op_type):
        latest = (DbOperation.select().where((DbOperation.op_type == op_type) &
                    (DbOperation.op_status == 'success'))
                  .order_by(DbOperation.op_time.desc())
                  .first())
        return latest

    def update_aliyun_binlog_files(self, binlog_files: List):
        DbAliyunBinlogFiles.insert_many(binlog_files).on_conflict_ignore().execute()

    def get_new_binlog_files(self) -> List:
        query = (DbAliyunBinlogFiles
                 .select()
                 .where((DbAliyunBinlogFiles.local_status == 'New') &
                        (DbAliyunBinlogFiles.remote_status == 'Completed'))
                 .order_by(DbAliyunBinlogFiles.log_start_time))

        return query

    def get_capital_data_by_month(self, month_str) -> List:
        query = (DbStockCapitalDataNew
                 .select(DbStockCapitalDataNew, DbStockCapitalDataNew.symbol_meta.alias('symbol_id'))
                 .where(fn.DATE_FORMAT(DbStockCapitalDataNew.date, '%Y%m') == month_str)
                 .order_by(DbStockCapitalDataNew.symbol)
                 .dicts())
        return list(query)
