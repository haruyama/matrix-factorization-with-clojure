(ns matrix-factorization.core-test
  (:require [clojure.test :refer :all]
            [matrix-factorization.core :refer :all]
            [incanter.core :refer [matrix]]))

(deftest norm-test
  (testing "matrix-factorization.core/norm"
    (are [x y] (= x (matrix-factorization.core/norm y))
         7.745966692414834 (incanter.core/matrix (range -4 5) 3))))

(deftest dot-test
  (testing "matrix-factorization.core/dot"
    (are [e v1 v2] (= e (matrix-factorization.core/dot v1 v2))
         11 [1 2] [3 4])))

(deftest update-matrix-in-test
  (testing "matrix-factorization.core/update-matrix-in"
    (are [x y] (= x y)
         (incanter.core/matrix [[1 0] [0 0]])
         (update-matrix-in (incanter.core/matrix [[0 0] [0 0]]) [0 0] inc)
         (incanter.core/matrix [[0 0] [0 2]])
         (update-matrix-in (incanter.core/matrix [[0 0] [0 0]]) [1 1] + 2)
         )))
