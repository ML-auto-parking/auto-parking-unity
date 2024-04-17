using System;
using System.Collections.Generic;

namespace AutonomousParking.Common.Extensions
{
    // SequenceExtensions 클래스는 컬렉션과 시퀀스 작업을 위한 확장 메서드(인덱스 관리, 요소 교환, 극값 검색 등)를 제공합니다.
    public static class SequenceExtensions
    {
        // 컬렉션의 첫 번째 인덱스를 반환합니다. ICollection<T>는 0부터 시작합니다.
        public static int FirstIndex<T>(this ICollection<T> source) => default;

        // 컬렉션의 마지막 인덱스를 반환합니다.
        public static int LastIndex<T>(this ICollection<T> source) => source.Count - 1;

        // 주어진 인덱스가 컬렉션의 범위 내에 있는지 확인합니다.
        public static bool IsInIndexRange<T>(this ICollection<T> source, int index) =>
            index >= source.FirstIndex() && index < source.LastIndex();

        // 인덱스를 다음으로 순환시키는 메서드입니다. 범위를 초과하는 경우 첫 인덱스로 돌아갑니다.
        public static int NextInCycledRange<T>(this int index, ICollection<T> collection) =>
            collection.IsInIndexRange(index) ? index + 1 : collection.FirstIndex();

        // 두 요소의 위치를 교환합니다. 위치가 동일하지 않을 경우에만 교환을 수행합니다.
        public static IList<T> SwapItems<T>(this IList<T> source, int firstIndex, int secondIndex)
        {
            source.ThrowExceptionIfArgumentIsNull(nameof(source)).ThrowExceptionIfNoElements();
            firstIndex.ThrowExceptionIfArgumentOutOfRange(nameof(firstIndex), source);
            secondIndex.ThrowExceptionIfArgumentOutOfRange(nameof(secondIndex), source);

            if (firstIndex != secondIndex)
                (source[firstIndex], source[secondIndex]) = (source[secondIndex], source[firstIndex]);
            return source;
        }

        // 시퀀스에서 keySelector를 사용해 최소 값을 찾는 메서드입니다.
        public static TSource MinBy<TSource, TKey>(this IEnumerable<TSource> source, Func<TSource, TKey> keySelector)
            where TKey : IComparable<TKey> => ExtremumBy(source, keySelector, (key1, key2) => key1.IsLessThan(key2));

        // 특정 조건(predicate)에 따라 시퀀스에서 극값(최소값 또는 최대값)을 찾습니다.
        public static TSource ExtremumBy<TSource, TKey>(this IEnumerable<TSource> source,
            Func<TSource, TKey> keySelector, Func<TKey, TKey, bool> predicate) where TKey : IComparable<TKey>
        {
            source.ThrowExceptionIfArgumentIsNull(nameof(source)).ThrowExceptionIfNoElements();
            keySelector.ThrowExceptionIfArgumentIsNull(nameof(keySelector));

            using IEnumerator<TSource> enumerator = source.GetEnumerator();

            (TSource Value, TKey Key) extremum = enumerator.IterateToFirstItemWithNotNullKey(keySelector);
            while (enumerator.MoveNext())
            {
                (TSource Value, TKey Key) item = (enumerator.Current, keySelector(enumerator.Current));
                if (item.Key != null && predicate(item.Key, extremum.Key))
                    extremum = item;
            }

            return extremum.Value;
        }

        // 반복자를 사용하여 keySelector에 의해 null이 아닌 첫 번째 키를 가진 요소를 찾습니다.
        public static (TSource Value, TKey Key) IterateToFirstItemWithNotNullKey<TSource, TKey>(
            this IEnumerator<TSource> enumerator, Func<TSource, TKey> keySelector) where TKey : IComparable<TKey>
        {
            enumerator.ThrowExceptionIfArgumentIsNull(nameof(enumerator));
            keySelector.ThrowExceptionIfArgumentIsNull(nameof(keySelector));

            enumerator.MoveNext();
            (TSource Value, TKey Key) item = (enumerator.Current, keySelector(enumerator.Current));
            if (default(TKey) is null)
                while (enumerator.MoveNext() && item.Key == null)
                    item = (enumerator.Current, keySelector(enumerator.Current));
            return item;
        }
    }
}
