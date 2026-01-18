# NEXUS - Arquitectura de Conexión entre Módulos

## Lead Software Architect & Behavioral Economist

---

## 📋 Resumen Ejecutivo

Este documento explica cómo `database_setup.py` y `economy_engine.py` se conectan para crear el sistema de incentivos conductuales que **fomenta retiros a Kraken Partner** mientras **desincentiva retiros a PayPal**.

---

## 🔗 Flujo de Conexión

### 1. Inicialización de la Base de Datos

```python
from database_setup import init_database, User, WithdrawalMethod
from economy_engine import NexusBank

# Crear base de datos y sesión
engine, SessionLocal = init_database("nexus_core.sqlite")
db = SessionLocal()
```

**¿Qué hace?**
- Crea el archivo SQLite `nexus_core.sqlite`
- Genera todas las tablas: `users`, `market_intel`, `predictions`, `user_bets`
- Retorna una sesión de SQLAlchemy para operaciones de BD

---

### 2. Creación del Motor Económico

```python
# Inicializar NexusBank con la sesión de BD
bank = NexusBank(db)
```

**¿Qué hace?**
- `NexusBank` recibe la sesión de BD como dependencia
- Puede leer/escribir en la tabla `users` para actualizar saldos
- Implementa toda la lógica de comisiones e incentivos

---

### 3. Flujo Completo: Usuario → Retiro → Incentivo

#### Paso A: Usuario tiene puntos virtuales

```python
# Usuario existe en BD con nexus_points = 100.0
user = db.query(User).filter(User.id == 1).first()
print(user.nexus_points)  # 100.0 puntos
```

#### Paso B: Usuario quiere retirar 80 puntos

```python
# Comparar métodos (diseño conductual)
comparison = compare_withdrawal_methods(80.0)
```

**Resultado de la comparación:**

| Método | Comisión | Bonus | Neto Recibido | Fiat |
|--------|----------|-------|---------------|------|
| **PayPal** | 10% (8 pts) | 0% | 72 puntos | $0.72 |
| **Kraken** | 0% | +5% (4 pts) | **84 puntos** | **$0.84** |

**Efecto psicológico:** El usuario ve claramente que Kraken es mejor.

#### Paso C: Procesar retiro a Kraken

```python
withdrawal = bank.process_withdrawal(
    user_id=1,
    method=WithdrawalMethod.KRAKEN_PARTNER,
    points_amount=80.0
)
```

**¿Qué sucede internamente?**

1. **Validación:**
   ```python
   # economy_engine.py línea ~120
   if user.nexus_points < points_amount:
       raise ValueError("Saldo insuficiente")
   ```

2. **Cálculo de comisión (Kraken = incentivo):**
   ```python
   # economy_engine.py línea ~60
   fee_amount = 0.0  # Sin comisión
   bonus_amount = 80.0 * 0.05  # +4 puntos (5%)
   net_amount = 80.0 + 4.0  # = 84 puntos
   ```

3. **Actualización de saldos:**
   ```python
   # economy_engine.py línea ~140
   user.nexus_points -= 80.0  # Descontar puntos retirados
   user.nexus_points += 4.0   # Agregar bonus
   user.fiat_balance += 0.84   # Agregar fiat ($0.84)
   db.commit()  # Guardar en BD
   ```

**Resultado final:**
- Usuario retiró 80 puntos
- Recibió 84 puntos (más de lo que retiró)
- Tiene $0.84 en saldo fiat
- Le quedan 24 puntos virtuales

---

## 🧠 Diseño Conductual: ¿Por qué funciona?

### 1. **Anclaje (Anchoring Effect)**
- PayPal muestra comisión del 10% → establece un "ancla" de costo
- Kraken muestra 0% → parece "gratis" en comparación

### 2. **Efecto de Default (Default Bias)**
- Kraken es la opción "recomendada" por el sistema
- Los usuarios tienden a elegir la opción por defecto

### 3. **Recompensa Variable (Variable Rewards)**
- El bonus del 5% es inesperado pero predecible
- Crea dopamina: "¡Recibí más de lo que esperaba!"

### 4. **Fricción Estratégica**
- PayPal: Mínimo $5.00 (alta fricción) + 10% comisión
- Kraken: Mínimo $2.00 (baja fricción) + 0% comisión + bonus

**Resultado:** Los usuarios naturalmente prefieren Kraken.

---

## 📊 Diagrama de Flujo

```
Usuario tiene 100 puntos
         ↓
Quiere retirar 80 puntos
         ↓
    ┌─────────┴─────────┐
    ↓                   ↓
PayPal              Kraken
(10% fee)          (0% fee + 5% bonus)
    ↓                   ↓
72 puntos          84 puntos
$0.72              $0.84
    ↓                   ↓
    └─────────┬─────────┘
              ↓
    Usuario elige Kraken
              ↓
    bank.process_withdrawal()
              ↓
    Actualiza BD (users table)
              ↓
    Usuario feliz (recibió más)
```

---

## 🔧 Puntos de Integración Clave

### 1. **database_setup.py → economy_engine.py**

```python
# database_setup.py define:
class User(Base):
    nexus_points = Column(Float, default=0.0)
    fiat_balance = Column(Float, default=0.0)

# economy_engine.py usa:
user.nexus_points -= points_amount
user.fiat_balance += fiat_amount
```

### 2. **Enums compartidos**

```python
# database_setup.py define:
class WithdrawalMethod(enum.Enum):
    PAYPAL = "PAYPAL"
    KRAKEN_PARTNER = "KRAKEN_PARTNER"

# economy_engine.py usa:
def calculate_withdrawal_fee(method: WithdrawalMethod, amount: float)
```

### 3. **Sesión de BD compartida**

```python
# database_setup.py crea:
SessionLocal = sessionmaker(...)

# economy_engine.py recibe:
bank = NexusBank(db_session)  # db_session es SessionLocal()
```

---

## 🎯 Conclusión

**La conexión entre módulos logra:**

1. ✅ **Desincentivo PayPal:** Alta comisión + mínimo alto = fricción psicológica
2. ✅ **Incentivo Kraken:** Sin comisión + bonus + mínimo bajo = flujo natural
3. ✅ **Persistencia:** Todos los cambios se guardan en SQLite
4. ✅ **Transparencia:** El usuario ve claramente por qué Kraken es mejor

**El usuario termina eligiendo Kraken no por coerción, sino por diseño conductual inteligente.**

---

## 🚀 Próximos Pasos

1. Ejecutar `demo_connection.py` para ver el flujo completo
2. Integrar `intel_manager.py` para análisis de noticias
3. Crear endpoints FastAPI para exponer la funcionalidad
4. Implementar sistema de predicciones con IA
