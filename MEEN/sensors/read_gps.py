import gps

session = gps.gps(mode=gps.WATCH_ENABLE)
try:
    while True:
        report = session.next()
        if report['class'] == 'TPV':
            print("Time:", getattr(report, 'time', 'n/a'))
            print("Latitude:", getattr(report, 'lat', 'n/a'))
            print("Longitude:", getattr(report, 'lon', 'n/a'))
except KeyboardInterrupt:
    print("Exiting...")
