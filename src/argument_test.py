import sys
import settings
settings.override_if_main(__name__, 2)
import pprint

def main():
    if len(sys.argv) < 2:
        print('Usage: <model_class> <settings_overrides (optional)>')
        exit()

    model_class_name = sys.argv[1]
    print(f"model_class_name = {model_class_name}")

    print("settings = ")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(settings.__settings)

if __name__ == "__main__":
    main()