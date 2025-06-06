from app import create_app

app = create_app('config.Config')

if __name__ == '__main__':
    print("Launching web ui in browser")
    app.run()