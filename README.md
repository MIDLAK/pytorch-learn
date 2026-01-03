# Требования
* python версии 3.14.2 и выше
* GPU с поддержкой CUDA (`torch.cuda.is_available()`) и 200 Гб свободного места на диске.

# Настройка Jupyter
1. `pip install ipykernel`
2. `ipython kernel install --name "pytorch" --user`

# Установка необходимых пакетов
Для установки пакетов нужно выполнить `pip install -r requirements.txt`. У меня были при первом запуске проблемы с проверочной хеш-суммой. Мне помог повторный запуск команды.

