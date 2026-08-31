Здесь описано, как подключить существующий или создать новый репозиторий, настроить аутентификацию по SSH и выполнить первый пуш. Также приведён типичный будничный цикл разработки, которого стоит придерживаться.

---

## 1. Клонирование существующего репозитория
Если репозиторий уже существует на GitHub (или другом сервере) и вы хотите получить его локальную копию, используйте команду `git clone`.
### Клонирование через SSH (рекомендуется)
```bash
git clone git@github.com:username/repository.git
```
### Клонирование через HTTPS 
```bash
git clone https://github.com/username/repository.git
```
После выполнения команды Git создаст папку с именем репозитория, скопирует всю историю, все ветки и автоматически настроит связь с удалённым репозиторием (origin). Вы сразу оказываетесь в ветке `main` (или `master`), и upstream уже настроен – можно сразу начинать работу.
### Что делать после клонирования

1. **Перейдите в папку проекта:**
```bash
cd repository
```

2. **Проверьте статус и настройки:**
```bash
git status
git remote -v   # покажет, что origin уже привязан
```

3. **Создайте ветку для своей задачи:**
```bash
git checkout -b feature/my-feature
```

4. **Работайте, делайте коммиты, пушите:**
```bash
git add .
git commit -m "feat: add something"
git push -u origin feature/my-feature   # первый пуш для новой ветки
```

5. **Периодически обновляйте свою ветку из `main`, чтобы избежать конфликтов:**
```bash 
git checkout main
git pull origin main        # обновить локальный main
git checkout feature/my-feature
git merge main              # или git rebase main
```
## 2. Создание нового репозитория (с нуля)
Если проект еще не начат и кода пока нет.
### Локальная часть
```bash
# 1. Создайте папку проекта
mkdir my-project
cd my-project

# 2. Инициализируйте Git
git init

# 3. Создайте файлы (например, README.md, .gitignore, код)
echo "# My Project" > README.md
echo "notes.json" > .gitignore

# 4. Добавьте файлы в индекс и сделайте первый коммит
git add .
git commit -m "Initial commit"
```
### Удалённая часть (GitHub)

1. Зайдите на GitHub, нажмите **New repository**.

2. Введите имя, **не создавайте** README, .gitignore или лицензию (они уже есть локально).

3. Нажмите **Create repository**.

4. Скопируйте SSH-адрес репозитория (например, `git@github.com:username/my-project.git`).

### Связывание и первый пуш

```bash
git remote add origin git@github.com:username/my-project.git
git branch -m master main   # если хотите использовать main вместо master
git push -u origin main
```

## 3. Ежедневный рабочий цикл

1. **Обновить `main`:**
```bash
git checkout main && git pull
```

2. **Создать ветку:** 
```
git checkout -b feature/...
```

3. **Написать код, закоммитить, запушить в свою ветку.**
 
4. **Создать PR на GitHub.**

5. **После слияния PR: обновить локальный `main` и удалить ветку.**