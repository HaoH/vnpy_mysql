import hashlib
from datetime import datetime, date
from typing import List, Dict

from peewee import (
    AutoField,
    CharField,
    DateTimeField,
    DoubleField,
    IntegerField,
    Model,
    MySQLDatabase as PeeweeMySQLDatabase,
    ModelSelect,
    ModelDelete,
    chunked,
    fn,
    DoubleField,
    DateField,
    BooleanField, BigIntegerField
)
from playhouse.shortcuts import ReconnectMixin

from vnpy.trader.constant import Exchange, Interval, Market
from vnpy.trader.object import BarData, TickData
from vnpy.trader.database import (
    BaseDatabase,
    BarOverview,
    TickOverview,
    DB_TZ,
    convert_tz
)
from vnpy.trader.setting import SETTINGS
from ex_vnpy.object import BasicStockData, BasicIndexData
from ex_vnpy.trade_plan import StoplossReason


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


class DbBasicStockData(Model):
    """股票列表映射对象"""

    id = AutoField()

    symbol: str = CharField()
    name: str = CharField()
    exchange: str = CharField()
    market: str = CharField()

    shares_total: float = DoubleField(null=True)
    shares_total_a: float = DoubleField(null=True)
    shares_circ_a: float = DoubleField(null=True)
    shares_non_circ_a: float = DoubleField(null=True)

    ex_date: date = DateField(null=True)

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

    hash_value: str = CharField()  # 对象的hash值
    update_dt: datetime = DateTimeField()

    class Meta:
        database: PeeweeMySQLDatabase = db
        indexes = ((("symbol", "exchange", "market"), True),)

    def has_changed(self, bs):
        sf_dict = self.__dict__
        keys = sf_dict['__data__'].keys()
        for key in keys:
            if key not in ['id', 'update_dt']:
                if getattr(self, key) != getattr(bs, key):
                    print('change: {}. old: {}, new: {}'.format(key, getattr(self, key), getattr(bs, key)))
                    return True
        return False

    @classmethod
    def get_hash_value(cls, sf_dict):
        # sf_dict = self.__dict__['__data__']
        for key in ['id', 'hash_value', 'update_dt']:
            if key in sf_dict.keys():
                sf_dict.pop(key)
        return hashlib.md5(sf_dict.__str__().encode('utf-8')).hexdigest()

    def save(self, *args, **kwargs):
        sf_dict = self.__dict__['__data__']
        self.hash_value = DbBasicStockData.get_hash_value(sf_dict)
        self.update_dt = datetime.now()
        return super(DbBasicStockData, self).save(*args, **kwargs)


class DbBasicIndexData(Model):
    """Index列表映射对象"""

    id = AutoField()

    symbol: str = CharField()
    name: str = CharField()
    full_name: str = CharField()
    exchange: str = CharField()
    market: str = CharField(default='CN')

    volume: int = BigIntegerField(null=True)
    turnover: int = BigIntegerField(null=True)

    publish_date: date = DateField(null=True)
    exit_date: date = DateField(null=True)
    has_price: bool = BooleanField(default=True)
    has_weight: bool = BooleanField(default=True)
    has_components: bool = BooleanField(default=True)

    class Meta:
        database: PeeweeMySQLDatabase = db
        indexes = ((("symbol", "exchange", "market"), True),)


class DbBacktestingResults(Model):
    """股票回测结果"""

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

    class Meta:
        database: PeeweeMySQLDatabase = db
        indexes = ((("symbol", "exchange", "market", "backtesting_dt"), True),)

    def save(self, *args, **kwargs):
        self.update_dt = datetime.now()
        return super(DbBacktestingResults, self).save(*args, **kwargs)


class MysqlDatabase(BaseDatabase):
    """Mysql数据库接口"""

    def __init__(self) -> None:
        """"""
        self.db: PeeweeMySQLDatabase = db
        self.db.connect()
        self.db.create_tables([DbBarData, DbTickData, DbBarOverview, DbTickOverview])

    def save_index_bar_data(self, bars: List[BarData], stream: bool = False) -> bool:
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

    def save_bar_data(self, bars: List[BarData], stream: bool = False) -> bool:
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

    def save_tick_data(self, ticks: List[TickData], stream: bool = False) -> bool:
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

    def get_bar_overview(self, type: str = "CS") -> List[BarOverview]:
        """查询数据库中的K线汇总信息"""
        # 如果已有K线，但缺失汇总信息，则执行初始化
        data_count: int = DbBarData.select().count() if type == "CS" else DbIndexBarData.select().count()
        overview_count: int = DbBarOverview.select().where((DbBarOverview.type == type)).count()
        if data_count and not overview_count:
            if type == "CS":
                self.init_bar_overview()
            else:
                self.init_index_bar_overview()

        s: ModelSelect = DbBarOverview.select().where((DbBarOverview.type == type))
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

    def get_basic_stock_data(self) -> Dict[Market, List[BasicStockData]]:
        """查询数据库中的基础信息汇总信息"""

        s: ModelSelect = DbBasicStockData.select()
        overviews = {}
        for overview in s:
            overview.exchange = Exchange(overview.exchange)
            market = Market(overview.market)
            overview.market = market
            if market not in overviews:
                overviews[market] = []
            overviews[overview.market].append(overview)
        return overviews

    def get_basic_index_data(self) -> Dict[Market, List[BasicIndexData]]:
        """查询数据库中的基础信息汇总信息"""

        s: ModelSelect = DbBasicIndexData.select()
        overviews = {}
        for overview in s:
            overview.exchange = Exchange(overview.exchange)
            market = Market(overview.market)
            overview.market = market
            if market not in overviews:
                overviews[market] = []
            overviews[overview.market].append(overview)
        return overviews

    def get_basic_info_by_symbol(self, symbol) -> BasicStockData:
        """查询数据库中的基础信息"""

        db_data: DbBasicStockData = (
            DbBasicStockData.select()
                .where(DbBasicStockData.symbol == symbol)
                .first()
        )
        dc = db_data.__data__
        for key in ("id", "hash_value"):
            dc.pop(key)

        dc["gateway_name"] = ""
        basic_data: BasicStockData = BasicStockData(**dc)
        return basic_data
